"""Stage and upload DustyLM model artifacts to Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi

try:
    from huggingface_hub.errors import HfHubHTTPError
except ImportError:  # pragma: no cover - compatibility with older hub releases
    HfHubHTTPError = Exception

from dustylm.config import get_profile, list_profiles
from dustylm.generate import resolve_generation_checkpoint_path
from dustylm.tokenizer import (
    CHATML_END_TOKEN,
    CHATML_START_TOKEN,
    END_OF_TEXT_TOKEN,
    SPECIAL_TOKENS,
)

try:
    from scripts.export_onnx import export_onnx
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from export_onnx import export_onnx

DEFAULT_PROFILE = "sft_dusty8m"
DEFAULT_MODEL_CARD = Path("artifacts/hf/HF_MODEL_CARD.md")
DEFAULT_LOGO = Path("docs/images/logo.png")
DEFAULT_COMMIT_MESSAGE = "Upload DustyLM model artifacts"
HUB_ONNX_FILENAME = "model_int8.onnx"
HUB_CHECKPOINT_FILENAME = "model.pt"
HUB_LOGO_FILENAME = Path("assets/logo.png")
# The Chat Template tells Hugging Face how to format conversations for this specific model.
# It uses Jinja syntax to loop through a list of messages and wrap them in our special tokens.
# Example: {"role": "user", "content": "hi"} becomes "<|im_start|>user\nhi<|im_end|>\n"
HF_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<|im_start|>assistant\\n'}}"
    "{% endif %}"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage and upload DustyLM artifacts to Hugging Face Hub."
    )
    parser.add_argument("--repo-id", required=True, help="Target model repo id.")
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        choices=list_profiles(),
        help="Profile whose checkpoint/tokenizer should be staged.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Checkpoint to stage. Defaults to the profile generation checkpoint.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Tokenizer JSON to stage. Defaults to the profile tokenizer.",
    )
    parser.add_argument(
        "--model-card",
        type=Path,
        default=DEFAULT_MODEL_CARD,
        help="Local Hugging Face model card copied to README.md.",
    )
    parser.add_argument(
        "--logo",
        type=Path,
        default=DEFAULT_LOGO,
        help="Logo image copied to assets/logo.png for the Hub README.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Temporary folder rebuilt before upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build staging files and stop before uploading.",
    )
    parser.add_argument(
        "--clean-after-upload",
        action="store_true",
        help="Delete staging files after a successful real upload.",
    )
    parser.add_argument(
        "--commit-message",
        default=DEFAULT_COMMIT_MESSAGE,
        help="Hugging Face commit message.",
    )
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Skip ONNX export (useful for base pretrain-only models).",
    )
    parser.add_argument(
        "--skip-chat-template",
        action="store_true",
        help="Skip chat template in tokenizer config (for base models).",
    )
    return parser.parse_args(argv)


def default_staging_dir(profile_name: str) -> Path:
    """Return the local staging directory rebuilt before Hub upload."""
    return Path("artifacts") / "hub_upload" / profile_name


def resolve_source_paths(
    profile_name: str,
    checkpoint_path: Path | None,
    tokenizer_path: Path | None,
) -> tuple[Path, Path]:
    """Resolve checkpoint and tokenizer paths from overrides or profile defaults."""
    profile = get_profile(profile_name)
    resolved_checkpoint_path = resolve_generation_checkpoint_path(
        profile,
        checkpoint_path=checkpoint_path,
    )
    resolved_tokenizer_path = Path(tokenizer_path or profile.model.tokenizer.path_or_name)
    return resolved_checkpoint_path, resolved_tokenizer_path


def require_file(path: Path, label: str) -> None:
    """Validate that a required source artifact exists and is a file."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")


def reset_staging_dir(staging_dir: Path) -> None:
    """Recreate a staging directory after basic safety checks."""
    resolved = staging_dir.resolve()
    if resolved == Path.cwd().resolve() or resolved == resolved.parent:
        raise ValueError(f"Refusing to clear unsafe staging directory: {staging_dir}")
    if len(resolved.parts) < 3:
        raise ValueError(f"Refusing to clear unsafe staging directory: {staging_dir}")

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)


def build_tokenizer_assets(tokenizer_path: Path, staging_dir: Path) -> None:
    """Create Hugging Face tokenizer files, including Dusty's chat template."""
    try:
        from transformers import PreTrainedTokenizerFast
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face tokenizer export requires transformers. "
            "Install with `uv sync --extra hub`, or run via `uv run --extra hub ...`."
        ) from exc

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    tokenizer.add_special_tokens(
        {
            "bos_token": CHATML_START_TOKEN,
            "eos_token": CHATML_END_TOKEN,
            "unk_token": END_OF_TEXT_TOKEN,
            "pad_token": END_OF_TEXT_TOKEN,
            "additional_special_tokens": SPECIAL_TOKENS,
        }
    )
    tokenizer.chat_template = HF_CHAT_TEMPLATE
    tokenizer.save_pretrained(staging_dir)
    write_tokenizer_metadata(staging_dir)


def write_tokenizer_metadata(staging_dir: Path) -> None:
    """Ensure tokenizer metadata includes chat template and special tokens."""
    tokenizer_config_path = staging_dir / "tokenizer_config.json"
    tokenizer_config = {}
    if tokenizer_config_path.exists():
        tokenizer_config = json.loads(tokenizer_config_path.read_text())
    tokenizer_config["chat_template"] = HF_CHAT_TEMPLATE
    tokenizer_config_path.write_text(json.dumps(tokenizer_config, indent=2, sort_keys=True) + "\n")

    special_tokens_map = {
        "bos_token": CHATML_START_TOKEN,
        "eos_token": CHATML_END_TOKEN,
        "unk_token": END_OF_TEXT_TOKEN,
        "pad_token": END_OF_TEXT_TOKEN,
        "additional_special_tokens": SPECIAL_TOKENS,
    }
    (staging_dir / "special_tokens_map.json").write_text(
        json.dumps(special_tokens_map, indent=2, sort_keys=True) + "\n"
    )


def stage_hub_artifacts(
    profile_name: str,
    checkpoint_path: Path,
    tokenizer_path: Path,
    model_card_path: Path,
    logo_path: Path,
    staging_dir: Path,
    skip_onnx: bool = False,
    skip_chat_template: bool = False,
) -> list[Path]:
    """Stage checkpoint, tokenizer assets, model card, logo, and optional ONNX."""
    require_file(checkpoint_path, "Checkpoint")
    require_file(tokenizer_path, "Tokenizer")
    require_file(model_card_path, "Model card")
    require_file(logo_path, "Logo")

    reset_staging_dir(staging_dir)

    shutil.copy2(checkpoint_path, staging_dir / HUB_CHECKPOINT_FILENAME)

    if not skip_onnx:
        export_onnx(
            profile_name=profile_name,
            checkpoint_step=None,
            checkpoint_path=checkpoint_path,
            output_path=staging_dir / HUB_ONNX_FILENAME,
            tokenizer_output_path=None,
            quantize=True,
            opset=23,
        )

    if skip_chat_template:
        shutil.copy2(tokenizer_path, staging_dir / "tokenizer.json")
    else:
        build_tokenizer_assets(tokenizer_path, staging_dir)

    shutil.copy2(model_card_path, staging_dir / "README.md")
    hub_logo_path = staging_dir / HUB_LOGO_FILENAME
    hub_logo_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(logo_path, hub_logo_path)

    return sorted(path for path in staging_dir.rglob("*") if path.is_file())


def is_auth_error(exc: Exception) -> bool:
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code in {401, 403}:
        return True
    message = str(exc).lower()
    return "token" in message or "unauthorized" in message or "forbidden" in message


def upload_staging_dir(
    staging_dir: Path,
    repo_id: str,
    commit_message: str,
) -> None:
    try:
        HfApi().upload_folder(
            folder_path=str(staging_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
    except HfHubHTTPError as exc:
        if is_auth_error(exc):
            raise RuntimeError(
                "Hugging Face upload is not authenticated. "
                "Set HF_TOKEN, run `huggingface-cli login`, or pass a valid token "
                "through your Hugging Face environment."
            ) from exc
        raise


def main(argv=None) -> None:
    args = parse_args(argv)
    checkpoint_path, tokenizer_path = resolve_source_paths(
        profile_name=args.profile,
        checkpoint_path=args.checkpoint_path,
        tokenizer_path=args.tokenizer_path,
    )
    staging_dir = args.staging_dir or default_staging_dir(args.profile)

    staged_files = stage_hub_artifacts(
        profile_name=args.profile,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        model_card_path=args.model_card,
        logo_path=args.logo,
        staging_dir=staging_dir,
        skip_onnx=args.skip_onnx,
        skip_chat_template=args.skip_chat_template,
    )
    print(f"Staged {len(staged_files)} files in {staging_dir}:")
    for path in staged_files:
        print(f"  {path.relative_to(staging_dir)}")

    if args.dry_run:
        print("Dry run complete. Skipping Hugging Face upload.")
        return

    upload_staging_dir(
        staging_dir=staging_dir,
        repo_id=args.repo_id,
        commit_message=args.commit_message,
    )
    print(f"Uploaded {staging_dir} to {args.repo_id}")

    if args.clean_after_upload:
        shutil.rmtree(staging_dir)
        print(f"Deleted staging directory {staging_dir}")


if __name__ == "__main__":
    main()
