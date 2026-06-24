"""Export a DustyLM profile checkpoint to ONNX for browser experiments.

The export is intentionally simple: one forward pass maps ``input_ids`` to
``logits``. Browser generation can call the graph repeatedly and keep the
current prompt window client-side.
"""

import argparse
import shutil
from pathlib import Path

import torch

from dustylm.config import get_profile, list_profiles
from dustylm.generate import resolve_generation_checkpoint_path
from dustylm.modeling import build_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Export DustyLM checkpoint to ONNX")
    parser.add_argument(
        "--profile",
        default="sft_dusty8m",
        choices=list_profiles(),
        help="Profile whose architecture/checkpoint should be exported.",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Export a step checkpoint instead of the profile's final checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Export a specific checkpoint path instead of the profile default.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/model.onnx"),
        help="Destination ONNX file.",
    )
    parser.add_argument(
        "--tokenizer-output",
        type=Path,
        default=Path("docs/tokenizer.json"),
        help="Destination tokenizer JSON copied next to the ONNX model.",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Keep the exported model as float32 instead of dynamic int8 quantization.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=23,
        help="ONNX opset version.",
    )
    return parser.parse_args(argv)


def load_state_dict(path: Path):
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    state_dict.pop("rope.sin_cache", None)
    state_dict.pop("rope.cos_cache", None)
    return state_dict


def quantize_onnx_model(output_path: Path) -> None:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise RuntimeError(
            "ONNX quantization requires onnxruntime. "
            "Install the ONNX extras with `uv sync --extra onnx`."
        ) from exc

    fp32_path = output_path.with_name(f"{output_path.stem}_fp32{output_path.suffix}")
    output_path.rename(fp32_path)
    quantize_dynamic(
        str(fp32_path),
        str(output_path),
        op_types_to_quantize=None,
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QInt8,
    )
    fp32_path.unlink()


def validate_onnx_model(output_path: Path, dummy_input: torch.Tensor) -> None:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX validation requires onnxruntime. "
            "Install the ONNX extras with `uv sync --extra onnx`."
        ) from exc

    session = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )
    session.run(["logits"], {"input_ids": dummy_input.cpu().numpy()})


def export_onnx(
    profile_name: str,
    checkpoint_step: int | None,
    output_path: Path,
    tokenizer_output_path: Path | None,
    quantize: bool,
    opset: int,
    checkpoint_path: Path | None = None,
) -> None:
    profile = get_profile(profile_name)
    checkpoint_path = resolve_generation_checkpoint_path(
        profile,
        checkpoint_step=checkpoint_step,
        checkpoint_path=checkpoint_path,
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    tokenizer_path = Path(profile.model.tokenizer.path_or_name)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    model = build_model(profile)
    model.load_state_dict(load_state_dict(checkpoint_path))
    model.eval()

    total_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Exporting {profile.name}: {total_params:,} params")
    print(f"Checkpoint: {checkpoint_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randint(
        low=0,
        high=profile.model.vocab_size,
        size=(1, min(32, profile.model.max_seq_len)),
        dtype=torch.long,
    )

    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        dynamo=True,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_shapes=({0: "batch_size", 1: "seq_len"},),
        opset_version=opset,
        external_data=False,
    )
    print(f"Exported {output_path} ({output_path.stat().st_size / 1_000_000:.1f} MB)")

    if quantize:
        quantize_onnx_model(output_path)
        print(
            f"Quantized {output_path} ({output_path.stat().st_size / 1_000_000:.1f} MB)"
        )

    validate_onnx_model(output_path, dummy_input)
    print(f"Validated ONNX Runtime inference for {output_path}")

    if tokenizer_output_path is not None:
        tokenizer_output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tokenizer_path, tokenizer_output_path)
        print(f"Copied tokenizer to {tokenizer_output_path}")


def main(argv=None):
    args = parse_args(argv)
    export_onnx(
        profile_name=args.profile,
        checkpoint_step=args.checkpoint_step,
        output_path=args.output,
        tokenizer_output_path=args.tokenizer_output,
        quantize=not args.no_quantize,
        opset=args.opset,
        checkpoint_path=args.checkpoint_path,
    )


if __name__ == "__main__":
    main()
