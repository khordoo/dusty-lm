"""Compare DustyLM step checkpoints on a fixed set of evaluation inputs.

By default, reports are written to ``artifacts/evaluations/checkpoints/`` as both
JSON and CSV files named like ``run_YYYYMMDD_HHMMSS_sft.json`` and
``run_YYYYMMDD_HHMMSS_sft.csv``. Use ``--output-dir`` or ``--run-id`` to
override those defaults.

Examples:
    uv run python evaluation/compare_checkpoints.py \
        --profile dusty8m \
        --input-set base \
        --steps 1000 3000 5700

    uv run python evaluation/compare_checkpoints.py \
        --profile sft_dusty8m \
        --input-set sft \
        --steps 900 19600
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dustylm.config import get_profile, list_profiles
from dustylm.generate import (
    encode_prompt,
    generate_token_ids,
    load_model,
    prepare_generation_prompt,
    validate_generation_options,
)

DEFAULT_OUTPUT_DIR = Path("artifacts/evaluations/checkpoints")
DEFAULT_INPUT_DIR = Path("evaluation/inputs")
INPUT_SETS = {"base", "sft"}


@dataclass(frozen=True)
class EvaluationInput:
    id: int
    category: str
    input: str


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate CSV and JSON comparisons across DustyLM checkpoints."
    )
    parser.add_argument(
        "--profile",
        required=True,
        choices=list_profiles(),
        help="Profile architecture/generation config to evaluate.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        required=True,
        help="Checkpoint step numbers to compare, such as 900 19600.",
    )
    parser.add_argument(
        "--input-set",
        choices=["auto", *sorted(INPUT_SETS)],
        default="auto",
        help="Named input set. Auto uses sft for profiles containing 'sft', otherwise base.",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        default=None,
        help="Custom JSON input file. Overrides --input-set path resolution.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated CSV and JSON reports.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run id used in output filenames.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Override profile top-p generation setting.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override profile temperature generation setting.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override profile max_new_tokens generation setting.",
    )
    return parser.parse_args(argv)


def infer_input_set(profile_name: str) -> str:
    if "sft" in profile_name:
        return "sft"
    return "base"


def resolve_input_path(
    profile_name: str,
    input_set: str,
    inputs_path: Path | None,
) -> tuple[str, Path]:
    if inputs_path is not None:
        if not inputs_path.exists():
            raise FileNotFoundError(f"Custom input file not found: {inputs_path}")
        resolved_set = input_set if input_set != "auto" else "custom"
        return resolved_set, inputs_path

    resolved_set = infer_input_set(profile_name) if input_set == "auto" else input_set
    path = DEFAULT_INPUT_DIR / f"{resolved_set}_inputs.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            f"The default input set for profile {profile_name!r} is {resolved_set!r}.\n"
            "Pass one of:\n"
            "  --input-set base\n"
            "  --input-set sft\n"
            "  --inputs path/to/custom_inputs.json"
        )
    print(f"Using {resolved_set} inputs: {path}")
    return resolved_set, path


def load_inputs(path: Path) -> list[EvaluationInput]:
    with path.open(encoding="utf-8") as file:
        raw_inputs = json.load(file)

    if not isinstance(raw_inputs, list):
        raise ValueError(f"Input file must contain a JSON list: {path}")

    seen_ids = set()
    inputs = []
    for index, item in enumerate(raw_inputs, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Input #{index} must be an object")
        missing = {"id", "category", "input"} - set(item)
        if missing:
            raise ValueError(f"Input #{index} missing fields: {sorted(missing)}")
        input_id = item["id"]
        if not isinstance(input_id, int):
            raise ValueError(f"Input #{index} id must be an integer")
        if input_id in seen_ids:
            raise ValueError(f"Duplicate input id: {input_id}")
        seen_ids.add(input_id)

        category = item["category"]
        user_input = item["input"]
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"Input #{index} category must be a non-empty string")
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError(f"Input #{index} input must be a non-empty string")
        inputs.append(
            EvaluationInput(
                id=input_id,
                category=category.strip(),
                input=user_input.strip(),
            )
        )

    if not inputs:
        raise ValueError(f"Input file is empty: {path}")
    return inputs


def build_run_id(input_set: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{timestamp}_{input_set}"


def generate_for_checkpoint(
    profile_name: str,
    step: int,
    inputs: list[EvaluationInput],
    top_p: float | None,
    temperature: float | None,
    max_new_tokens: int | None,
) -> list[dict[str, Any]]:
    profile = get_profile(profile_name)
    if profile.generation is None:
        raise ValueError(f"Profile {profile_name!r} does not define generation config")

    spec = profile.generation
    resolved_top_p = spec.top_p if top_p is None else top_p
    resolved_temperature = spec.temperature if temperature is None else temperature
    validate_generation_options(resolved_top_p, resolved_temperature)

    model, tokenizer, device = load_model(profile, checkpoint_step=step)
    rows = []
    for item in inputs:
        prompt = prepare_generation_prompt(item.input, profile)
        token_ids = encode_prompt(tokenizer, prompt, spec)
        result = generate_token_ids(
            model=model,
            tokenizer=tokenizer,
            token_ids=token_ids,
            spec=spec,
            max_seq_len=profile.model.max_seq_len,
            device=device,
            max_new_tokens=max_new_tokens,
            top_p=resolved_top_p,
            temperature=resolved_temperature,
        )
        rows.append(
            {
                "checkpoint_step": step,
                "input_id": item.id,
                "category": item.category,
                "input": item.input,
                "output": result.text.strip(),
                "finish_reason": result.finish_reason,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": len(result.token_ids),
            }
        )
    return rows


def write_json_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def write_csv_report(
    path: Path,
    rows: list[dict[str, Any]],
    run_metadata: dict[str, Any],
) -> None:
    fieldnames = [
        "run_id",
        "profile",
        "input_set",
        "checkpoint_step",
        "input_id",
        "category",
        "input",
        "output",
        "finish_reason",
        "prompt_tokens",
        "completion_tokens",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "run_id": run_metadata["run_id"],
                    "profile": run_metadata["profile"],
                    "input_set": run_metadata["input_set"],
                    **row,
                }
            )


def run_comparison(args) -> tuple[Path, Path]:
    """Run the full checkpoint comparison and write JSON/CSV reports.

    The parsed CLI args decide which profile, checkpoint steps, and input set to
    use. This function resolves the input file, generates outputs for every
    checkpoint/input pair, writes both report formats, and returns their paths.
    """
    input_set, input_path = resolve_input_path(
        profile_name=args.profile,
        input_set=args.input_set,
        inputs_path=args.inputs,
    )
    inputs = load_inputs(input_path)
    run_id = args.run_id or build_run_id(input_set)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for step in args.steps:
        print(f"Evaluating checkpoint step {step} on {len(inputs)} inputs...")
        results.extend(
            generate_for_checkpoint(
                profile_name=args.profile,
                step=step,
                inputs=inputs,
                top_p=args.top_p,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
        )

    metadata = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "profile": args.profile,
        "input_set": input_set,
        "input_path": str(input_path),
        "checkpoint_steps": args.steps,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }
    report = {
        **metadata,
        "inputs": [asdict(item) for item in inputs],
        "results": results,
    }

    json_path = args.output_dir / f"{run_id}.json"
    csv_path = args.output_dir / f"{run_id}.csv"
    write_json_report(json_path, report)
    write_csv_report(csv_path, results, metadata)
    return json_path, csv_path


def main(argv=None) -> None:
    args = parse_args(argv)
    json_path, csv_path = run_comparison(args)
    print(f"Saved JSON report: {json_path}")
    print(f"Saved CSV report:  {csv_path}")


if __name__ == "__main__":
    main()
