from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).parents[1]
TOKENIZER_NAME = "r50k_base"
IGNORE_INDEX = -100


@dataclass(frozen=True)
class ModelConfig:
    max_seq_len: int = 512
    embed_dim: int = 512
    num_heads: int = 8
    num_kv_heads: int = 2
    num_layers: int = 6
    rope_base: int = 10000
    rms_eps: float = 1e-4


@dataclass(frozen=True)
class TrainingConfig:
    max_seq_len: int = 1024
    learning_rate: float = 1e-4
    batch_size: int = 8
    raw_python_dataset_path: str = "../data/python_dataset"
    dataset_path: str = "data/tiny_codes_python_tokenized"
    checkpoint_path: Path = REPO_ROOT / "checkpoints" / "tinygpt_epoch_1.pt"
    log_dir: str = "runs/"


@dataclass(frozen=True)
class GenerationConfig:
    max_seq_len: int = 512
    max_new_tokens: int = 1000
    temperature: float = 1.0
    top_k: int = 10
    checkpoint_path: Path = REPO_ROOT / "checkpoints" / "tinygpt_epoch_1.pt"
    eos_text: str = "<|im_end|>"


MODEL_CONFIG = ModelConfig()
TRAINING_CONFIG = TrainingConfig()
GENERATION_CONFIG = GenerationConfig()
