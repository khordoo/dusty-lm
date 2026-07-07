"""Test different phrasings for the being_ignored topic."""

from pathlib import Path

from dustylm.checkpoint import resolve_profile_name_for_checkpoint
from dustylm.config import get_profile
from dustylm.generate import (
    encode_prompt,
    generate_token_ids,
    load_model,
    prepare_generation_prompt,
)

STEP = 200
TEMP = 0.6
RUNS = 5

VARIATIONS = [
    "does anyone notice when you clean?",
    "does anyone appreciate your work?",
    "do you feel ignored when no one sees you clean?",
    "do people ever thank you for cleaning?",
    "does it bother you when no one notices your work?",
    "nobody said anything about your cleaning. how do you feel?",
]


ckpt = Path("artifacts/checkpoints") / f"dusty8m_sft_step_{STEP}.pt"
pname = resolve_profile_name_for_checkpoint(ckpt, explicit_profile=None, default_profile="sft_dusty8m", mode="generation")
profile = get_profile(pname)
model, tokenizer, device = load_model(profile, checkpoint_path=ckpt)
spec = profile.generation

print(f"Step {STEP} @ temp {TEMP} | {RUNS} runs each\n")

for q in VARIATIONS:
    print(f"{'='*60}")
    print(f"QUESTION: {q}")
    print(f"{'='*60}")
    for run in range(RUNS):
        prompt = prepare_generation_prompt(q, profile)
        token_ids = encode_prompt(tokenizer, prompt, spec)
        result = generate_token_ids(
            model, tokenizer, token_ids, spec,
            max_seq_len=profile.model.max_seq_len,
            device=device,
            temperature=TEMP,
            top_p=0.8,
            top_k=5,
        )
        text = result.text.strip()
        print(f"  [{run+1}] {text[:80]}")
    print()
