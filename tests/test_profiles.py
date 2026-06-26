import pytest

from dustylm.config import (
    ModelFamily,
    ModelSpec,
    Profile,
    TokenizerSpec,
    TrainingTask,
    get_profile,
    list_profiles,
)
from dustylm.modeling import build_model
from dustylm.models.scratch import DustyLM as ScratchDustyLM
from dustylm.models.smollm2 import DustyLM as SmolLM2DustyLM


def test_profile_lookup_and_unknown_profile_errors():
    assert get_profile("scratch_small").name == "scratch_small"

    with pytest.raises(KeyError, match="Unknown profile"):
        get_profile("missing")


def test_scratch_training_profile_defines_optimizer_hyperparameters():
    profile = get_profile("scratch_small")

    assert profile.training is not None
    assert profile.training.task == TrainingTask.PRETRAIN
    assert profile.training.learning_rate == 1e-4
    assert profile.training.weight_decay == 0.01


def test_dusty8m_profile_is_scratch_pretrain_profile():
    profile = get_profile("dusty8m")

    assert profile.model.family == ModelFamily.SCRATCH_GPT
    assert profile.model.vocab_size == 4096
    assert profile.model.hidden_dim == 1024
    assert profile.training is not None
    assert profile.training.task == TrainingTask.PRETRAIN
    assert profile.training.output_checkpoint.name == "dusty8m.pt"
    assert profile.training.checkpoint_every_steps == 100
    assert profile.training.checkpoint_dir.name == "checkpoints"
    assert profile.generation is not None
    assert profile.generation.checkpoint_path.name == "dusty8m.pt"
    assert profile.generation.top_p == 0.9
    assert profile.generation.eos_text == "<|endoftext|>"


def test_sft_dusty8m_profile_finetunes_dusty_checkpoint_separately():
    pretrain = get_profile("dusty8m")
    sft = get_profile("sft_dusty8m")

    assert sft.model == pretrain.model
    assert sft.training is not None
    assert sft.training.task == TrainingTask.SFT
    assert sft.training.raw_sft_path.name == "dusty_sft.jsonl"
    assert sft.training.sft_user_field == "user"
    assert sft.training.sft_assistant_field == "dusty"
    assert sft.training.init_checkpoint_path.name == "dusty8m.pt"
    assert sft.training.output_checkpoint.name == "dusty8m_sft.pt"
    assert sft.training.checkpoint_every_steps == 100
    assert sft.training.checkpoint_dir.name == "checkpoints"
    assert sft.generation is not None
    assert sft.generation.checkpoint_path.name == "dusty8m_sft.pt"
    assert sft.generation.eos_token_id == 2
    assert sft.generation.top_p == 0.8
    assert sft.generation.eos_text is None
    assert sft.generation.max_chat_turns == 1


def test_smollm2_profiles_share_tokenizer_path():
    profiles = [
        get_profile(name)
        for name in list_profiles()
        if get_profile(name).model.family == ModelFamily.SMOLLM2
    ]

    tokenizer_paths = {profile.model.tokenizer.path_or_name for profile in profiles}

    assert tokenizer_paths == {
        get_profile("smollm2_360m").model.tokenizer.path_or_name
    }


def test_sft_profile_uses_base_model_spec():
    base = get_profile("smollm2_135m")
    sft = get_profile("sft_smollm2_135m")

    assert sft.base_profile == "smollm2_135m"
    assert sft.model == base.model
    assert sft.training is not None
    assert sft.generation is not None
    assert sft.generation.max_chat_turns == 5
    assert sft.hf_artifacts == base.hf_artifacts


def test_smollm2_profiles_define_download_artifacts():
    profile_360m = get_profile("smollm2_360m")
    profile_135m = get_profile("smollm2_135m")

    assert profile_360m.hf_artifacts is not None
    assert profile_360m.hf_artifacts.repo_id == "HuggingFaceTB/SmolLM2-360M"
    assert profile_360m.hf_artifacts.weights_filename == "model.safetensors"
    assert profile_360m.hf_artifacts.local_weights_path.name == "smollm2_360.safetensors"

    assert profile_135m.hf_artifacts is not None
    assert profile_135m.hf_artifacts.repo_id == "HuggingFaceTB/SmolLM2-135M"
    assert (
        profile_135m.hf_artifacts.local_weights_path.name
        == "smollm2_135m.safetensors"
    )
    assert profile_135m.model.hidden_dim == 1536


def test_build_model_dispatches_scratch_family():
    profile = Profile(
        name="tiny_scratch",
        model=ModelSpec(
            family=ModelFamily.SCRATCH_GPT,
            max_seq_len=16,
            vocab_size=32,
            embed_dim=16,
            num_heads=4,
            num_kv_heads=2,
            num_layers=1,
            hidden_dim=24,
            tokenizer=TokenizerSpec(kind="tiktoken", path_or_name="r50k_base"),
        ),
    )

    model = build_model(profile)

    assert isinstance(model, ScratchDustyLM)
    assert model.layers[0].mlp[0].out_features == 24


def test_build_model_dispatches_smollm2_family():
    profile = Profile(
        name="tiny_smollm2",
        model=ModelSpec(
            family=ModelFamily.SMOLLM2,
            max_seq_len=16,
            vocab_size=32,
            embed_dim=16,
            num_heads=4,
            num_kv_heads=2,
            num_layers=1,
            hidden_dim=32,
            tokenizer=TokenizerSpec(kind="tokenizers", path_or_name="tokenizer.json"),
        ),
    )

    assert isinstance(build_model(profile), SmolLM2DustyLM)
