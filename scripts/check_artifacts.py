"""Pre-flight check: verify profile artifacts exist before running inference.
Called from Makefile to produce a clean error (no traceback).
"""

import argparse
import sys
from pathlib import Path

from dustylm.checkpoint import CHAT_PROFILE_DEFAULT
from dustylm.config import get_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", nargs="?", default=CHAT_PROFILE_DEFAULT)
    parser.add_argument("mode", nargs="?", default="generate")
    parser.add_argument("--checkpoint-step", type=int, default=None)
    args = parser.parse_args()

    p = get_profile(args.profile)
    missing = []

    if args.checkpoint_step is not None:
        step_checkpoint = p.generation.checkpoint_path.parent / f"{p.generation.checkpoint_path.stem}_step_{args.checkpoint_step}.pt"
        if not step_checkpoint.exists():
            missing.append(str(step_checkpoint))
    elif p.generation and not Path(p.generation.checkpoint_path).exists():
        missing.append(str(p.generation.checkpoint_path))

    if p.model.tokenizer.kind != "tiktoken":
        tok_path = Path(p.model.tokenizer.path_or_name)
        if not tok_path.exists():
            missing.append(str(tok_path))

    if missing:
        print("Missing artifacts:")
        for m in missing:
            print(f"  {m}")
        print()
        if args.mode == "chat":
            print("Run 'make download-models' to download pre-trained weights,")
            print("or 'make train-end-to-end' to train from scratch.")
        else:
            print("Run 'make download-models' to download pre-trained weights,")
            print("or 'make train-end-to-end' to train from scratch.")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
