import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file

from tiny_gpt.config import REPO_ROOT, Profile, get_profile, list_profiles
from tiny_gpt.modeling import build_model


DEFAULT_HF_MODEL_PATHS = {
    "smollm2_360m": [
        REPO_ROOT / "artifacts" / "hf" / "smollm2_360m.safetensors",
        REPO_ROOT / "artifacts" / "hf" / "smollm2_360.safetensors",
    ],
}


def map_smollm2_key(hf_key: str) -> str:
    if hf_key == "model.embed_tokens.weight":
        return "embed_tokens.weight"
    if hf_key == "model.norm.weight":
        return "final_norm.weight"
    if hf_key == "lm_head.weight":
        return "vocab_proj.weight"
    return hf_key.replace("model.", "").replace(".mlp", "").replace(
        "o_proj", "out_proj"
    )


def convert_smollm2_state_dict(hf_weights: dict[str, torch.Tensor]):
    state_dict = {
        map_smollm2_key(hf_key): tensor for hf_key, tensor in hf_weights.items()
    }
    state_dict["vocab_proj.weight"] = state_dict["embed_tokens.weight"]
    return state_dict


def resolve_hf_model_path(profile: Profile, hf_model_path: str | Path | None) -> Path:
    if hf_model_path is not None:
        return Path(hf_model_path)

    if profile.hf_artifacts is not None and profile.hf_artifacts.local_weights_path.exists():
        return profile.hf_artifacts.local_weights_path

    candidates = DEFAULT_HF_MODEL_PATHS.get(profile.name, [])
    for candidate in candidates:
        if candidate.exists():
            if candidate != candidates[0]:
                print("Using discovered Hugging Face model file:", candidate)
            return candidate

    if candidates:
        return candidates[0]
    if profile.hf_artifacts is not None:
        return profile.hf_artifacts.local_weights_path
    return REPO_ROOT / "artifacts" / "hf" / f"{profile.name}.safetensors"


def map_smollm2_to_tinygpt_and_save(
    profile: Profile,
    hf_model_path: str | Path | None,
    tiny_gpt_save_path: str | Path | None = None,
):
    if profile.generation is None:
        raise ValueError(f"Profile '{profile.name}' does not define generation config")

    hf_model_path = resolve_hf_model_path(profile, hf_model_path)
    tiny_gpt_save_path = Path(tiny_gpt_save_path or profile.generation.checkpoint_path)

    print("Loading Hugging Face model weights from:", hf_model_path)
    hf_weights = load_file(str(hf_model_path))
    state_dict = convert_smollm2_state_dict(hf_weights)

    print("Validating converted state dict against TinyGPT model")
    model = build_model(profile)
    model.load_state_dict(state_dict)

    tiny_gpt_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), tiny_gpt_save_path)
    print("Saved converted TinyGPT checkpoint to:", tiny_gpt_save_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="smollm2_360m",
        choices=list_profiles(),
        help="Registered SmolLM2 profile to convert.",
    )
    parser.add_argument(
        "--hf-model-path",
        type=Path,
        default=None,
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    map_smollm2_to_tinygpt_and_save(
        profile=get_profile(args.profile),
        hf_model_path=args.hf_model_path,
        tiny_gpt_save_path=args.output,
    )


if __name__ == "__main__":
    main()
