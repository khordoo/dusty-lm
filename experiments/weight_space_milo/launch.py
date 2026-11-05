"""Optional separately billed text-only Responses transport. Never called by verify."""

import argparse
import json
import math
import os
import time
from pathlib import Path

from .controller import PROTOCOL, Controller, initialize, write_json
from .datasets import DEVELOPMENT
from .source import sha256

MAX_OUTPUT_TOKENS = 32768
MAX_INPUT_TOKENS = 100000


def request_options(messages, model="gpt-6-astra", max_output_tokens=MAX_OUTPUT_TOKENS):
    return dict(
        model=model,
        reasoning={"effort": "high", "context": "all_turns"},
        input=messages,
        tools=[],
        tool_choice="none",
        max_output_tokens=max_output_tokens,
        include=["reasoning.encrypted_content"],
        store=False,
    )


def assert_text_only(response):
    if response.status != "completed":
        raise ValueError("Editor response incomplete; no partial request execution")
    for item in response.output:
        if item.type not in {"message", "reasoning"}:
            raise ValueError(f"Unexpected editor tool/event: {item.type}")


def validate_launch_setup():
    from .source import ARTIFACTS, verify
    from .verify import infrastructure_hashes
    from .workflows import baseline

    verify()
    path = ARTIFACTS / "preflight.json"
    if not path.is_file():
        raise ValueError("Run make milo-verify before launching")
    report = json.loads(path.read_text())
    if (
        not report.get("tests", {}).get("ok")
        or not report.get("fresh_process_inference", {}).get("ok")
        or report.get("infrastructure_sha256") != infrastructure_hashes()
    ):
        raise ValueError("Full verification missing/stale; run make milo-verify")
    baseline()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--model", required=True, help="Explicit account-supported Astra model ID; no substitutions"
    )
    parser.add_argument("--billing", choices=["api"], required=True)
    parser.add_argument("--spend-limit-usd", type=float, required=True)
    parser.add_argument("--input-usd-per-million", type=float, required=True)
    parser.add_argument("--output-usd-per-million", type=float, required=True)
    parser.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    parser.add_argument("--execute", action="store_true", help="Explicit paid-run switch")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Refusing paid run without --execute; verification never calls editor")
    if not 1024 <= args.max_output_tokens <= 65536:
        raise SystemExit("Output cap must be 1024..65536 tokens")
    if "astra" not in args.model.lower():
        raise SystemExit("Explicit Astra model required; no model fallback")
    rates = [args.spend_limit_usd, args.input_usd_per_million, args.output_usd_per_million]
    if not all(math.isfinite(x) and x > 0 for x in rates):
        raise SystemExit("Positive finite ceiling and verified upper-bound token prices required")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY required only in host transport; never mounted/passed to tensor worker"
        )
    validate_launch_setup()
    from openai import OpenAI

    state = initialize(args.run)
    prompt = Path(__file__).with_name("editor_prompt.md").read_text()
    development = json.loads(DEVELOPMENT.read_text())
    import torch
    from tokenizers import Tokenizer

    tensors = torch.load(Path(state["source"]) / "model.pt", weights_only=True, map_location="cpu")
    if "model_state_dict" in tensors:
        tensors = tensors["model_state_dict"]
    tokenizer = Tokenizer.from_file(str(Path(state["source"]) / "tokenizer.json"))
    metadata = {
        "tensors": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in tensors.items()
        },
        "name_tokenization": {
            name: {"ids": tokenizer.encode(name).ids, "tokens": tokenizer.encode(name).tokens}
            for name in [" dusty", " milo", "dusty", "milo"]
        },
    }
    messages = [
        {"role": "developer", "content": prompt},
        {
            "role": "user",
            "content": json.dumps(
                {"protocol": PROTOCOL, "development": development, "source_metadata": metadata}
            ),
        },
    ]
    settings = {
        "model": args.model,
        "reasoning": "high",
        "tools": [],
        "tool_choice": "none",
        "transport": "Responses API; separate billing",
        "spend_limit_usd": args.spend_limit_usd,
        "input_usd_per_million": args.input_usd_per_million,
        "output_usd_per_million": args.output_usd_per_million,
        "max_output_tokens": args.max_output_tokens,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "prompt_sha256": sha256(Path(__file__).with_name("editor_prompt.md")),
        "started_at": time.time(),
    }
    write_json(args.run / "editor_settings.json", settings)
    import openai

    settings["openai_sdk_version"] = openai.__version__
    write_json(args.run / "editor_settings.json", settings)
    # No automatic SDK retries: a failed billed request must not be repeated invisibly.
    client = OpenAI(max_retries=0, timeout=300)
    controller = Controller(args.run)
    spent = 0.0
    for turn in range(180):
        # Conservative byte upper bound (UTF-8 bytes plus generous framing reserve).
        input_bound = len(json.dumps(messages, ensure_ascii=False).encode()) + 8192
        if input_bound > MAX_INPUT_TOKENS:
            raise SystemExit("Input token upper bound exhausted; no hidden compaction/new session")
        reservation = (
            input_bound * args.input_usd_per_million
            + args.max_output_tokens * args.output_usd_per_million
        ) / 1e6
        if spent + reservation > args.spend_limit_usd:
            raise SystemExit("Cannot reserve next call within spending ceiling; run incomplete")
        response = client.responses.create(
            **request_options(messages, args.model, args.max_output_tokens)
        )
        usage = response.usage
        spent += (
            usage.input_tokens * args.input_usd_per_million
            + usage.output_tokens * args.output_usd_per_million
        ) / 1e6
        event = {
            "turn": turn,
            "response": response.model_dump(mode="json"),
            "estimated_usd_upper_rates": spent,
        }
        with (args.run / "editor_transcript.jsonl").open("a") as log:
            log.write(json.dumps(event) + "\n")
        assert_text_only(response)
        text = response.output_text
        messages.extend(item.model_dump(mode="json", exclude_none=True) for item in response.output)
        try:
            request = json.loads(text)
            result = controller.dispatch(request)
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            result = {"error": str(error)}
        messages.append({"role": "user", "content": json.dumps(result)})
        if result.get("final"):
            print(json.dumps({"final": result["final"], "estimated_usd_upper_rates": spent}))
            return
    raise SystemExit("Controller turn ceiling exhausted; run incomplete")


if __name__ == "__main__":
    main()
