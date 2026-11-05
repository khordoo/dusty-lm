"""Download immutable experiment inputs, never ordinary checkpoint paths."""

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/weight_space_milo"
SOURCE = ARTIFACTS / "source"
REVISION = "43d341f16f17cbeefedfe2475582942ab5891ed1"
COMMIT = "7a624c91611daa06580bb9104d52e8013c47e114"
HASHES = {
    "model.pt": "5d7aa63d2db6d51f38e5e9c25a76facd33c0f84477ad940b97bc6ea6a1e661e5",
    "tokenizer.json": "989d4b2c4f65d408888261f7d412c8f64ea84a28e124b74ab3a8025e5a92dafe",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(directory=SOURCE):
    directory = Path(directory)
    for name, expected in HASHES.items():
        path = directory / name
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            raise ValueError(f"Pinned artifact verification failed: {path}")
    return {
        "repo": "mkhordoo/dusty-8m-sft",
        "revision": REVISION,
        "source_commit": COMMIT,
        "sha256": HASHES,
    }


def download(directory=SOURCE):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in HASHES.items():
        path = directory / name
        if path.exists():
            if path.is_symlink() or sha256(path) != expected:
                raise ValueError(f"Refusing to replace unexpected existing source: {path}")
            continue
        temporary = path.with_suffix(path.suffix + ".download")
        try:
            urllib.request.urlretrieve(
                f"https://huggingface.co/mkhordoo/dusty-8m-sft/resolve/{REVISION}/{name}", temporary
            )
            if sha256(temporary) != expected:
                raise ValueError(f"Download hash mismatch: {name}")
            os.replace(temporary, path)
            path.chmod(0o444)
        finally:
            temporary.unlink(missing_ok=True)
    manifest = verify(directory)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    print(json.dumps(verify() if args.verify else download(), indent=2))
