"""CLI for downloading HuggingFace model weights and tokenizer files.

Downloads are resolved from the ``HFArtifactSpec`` on each profile, cached
locally under ``artifacts/``, and optionally converted to DustyLM checkpoint
format in one step via the ``--convert`` flag.
"""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from dustylm.adapter import map_smollm2_to_dustylm_and_save
from dustylm.config import HFArtifactSpec, Profile, get_profile, list_profiles


def list_downloadable_profiles() -> list[str]:
    """Return base profiles that define downloadable Hugging Face artifacts."""
    return [
        name
        for name in list_profiles()
        if get_profile(name).hf_artifacts is not None and get_profile(name).base_profile is None
    ]


def copy_hf_file(
    artifacts: HFArtifactSpec,
    filename: str,
    destination: Path,
    force: bool = False,
) -> Path:
    """Download one Hub file and copy it into the repo's artifact layout."""
    destination = Path(destination)
    if destination.exists() and not force:
        print("exists:", destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded_path = hf_hub_download(
        repo_id=artifacts.repo_id,
        filename=filename,
        revision=artifacts.revision,
    )
    shutil.copyfile(downloaded_path, destination)
    print("downloaded:", destination)
    return destination


def download_profile_artifacts(
    profile: Profile,
    convert: bool = False,
    force: bool = False,
) -> None:
    """Download all Hub artifacts for a profile and optionally convert weights."""
    if profile.hf_artifacts is None:
        raise ValueError(f"Profile '{profile.name}' does not define HF artifacts")

    artifacts = profile.hf_artifacts
    weights_path = copy_hf_file(
        artifacts=artifacts,
        filename=artifacts.weights_filename,
        destination=artifacts.local_weights_path,
        force=force,
    )
    copy_hf_file(
        artifacts=artifacts,
        filename=artifacts.tokenizer_filename,
        destination=artifacts.local_tokenizer_path,
        force=force,
    )

    if convert:
        map_smollm2_to_dustylm_and_save(
            profile=profile,
            hf_model_path=weights_path,
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Download profile artifacts into the local artifacts/ folder."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument(
        "--profile",
        required=True,
        choices=list_downloadable_profiles(),
        help="Registered profile to download artifacts for.",
    )
    download_parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert downloaded HF safetensors into a DustyLM checkpoint.",
    )
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite local artifact files if they already exist.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "download":
        download_profile_artifacts(
            profile=get_profile(args.profile),
            convert=args.convert,
            force=args.force,
        )


if __name__ == "__main__":
    main()
