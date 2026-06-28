"""Pre-flight check: verify profile artifacts exist before running inference.
Called from Makefile to produce a clean error (no traceback).
"""

import sys
from pathlib import Path

from dustylm.checkpoint import CHAT_PROFILE_DEFAULT
from dustylm.config import get_profile


def main():
    profile_name = sys.argv[1] if len(sys.argv) > 1 else ""
    mode = sys.argv[2] if len(sys.argv) > 2 else "generate"

    if not profile_name:
        profile_name = CHAT_PROFILE_DEFAULT

    p = get_profile(profile_name)
    missing = []

    if p.generation and not Path(p.generation.checkpoint_path).exists():
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
        if mode == "chat":
            print("Run 'make download-models' to download pre-trained weights,")
            print("or 'make train-sft EPOCHS=23' to train from scratch.")
        else:
            print("Run 'make download-models' to download pre-trained weights,")
            print("or 'make train-pretrain EPOCHS=23' to train from scratch.")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
