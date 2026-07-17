from types import SimpleNamespace

import pytest
import torch

from scripts import export_onnx


class TinyExportModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 4)
        self.output = torch.nn.Linear(4, 16, bias=False)

    def forward(self, input_ids):
        return self.output(self.embedding(input_ids))


def build_separated_weights(model):
    with torch.no_grad():
        for token_id in range(16):
            model.embedding.weight[token_id].fill_(token_id + 1)
            model.output.weight[token_id].fill_((token_id + 1) / 10)


@pytest.mark.filterwarnings("ignore:.*LeafSpec.*:FutureWarning")
def test_exported_onnx_matches_pytorch_top_tokens(monkeypatch, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")

    checkpoint_path = tmp_path / "tiny.pt"
    tokenizer_path = tmp_path / "tokenizer.json"
    output_path = tmp_path / "tiny.onnx"
    tokenizer_output_path = tmp_path / "exported-tokenizer.json"

    reference_model = TinyExportModel().eval()
    build_separated_weights(reference_model)
    torch.save(reference_model.state_dict(), checkpoint_path)
    tokenizer_path.write_text("{}")

    profile = SimpleNamespace(
        name="tiny_export",
        model=SimpleNamespace(
            vocab_size=16,
            max_seq_len=8,
            tokenizer=SimpleNamespace(path_or_name=tokenizer_path),
        ),
    )
    monkeypatch.setattr(export_onnx, "get_profile", lambda name: profile)
    monkeypatch.setattr(
        export_onnx,
        "resolve_generation_checkpoint_path",
        lambda *args, **kwargs: checkpoint_path,
    )
    monkeypatch.setattr(export_onnx, "build_model", lambda profile: TinyExportModel())

    export_onnx.export_onnx(
        profile_name="tiny_export",
        checkpoint_step=None,
        checkpoint_path=None,
        output_path=output_path,
        tokenizer_output_path=tokenizer_output_path,
        quantize=True,
        opset=23,
    )

    input_ids = torch.tensor([[1, 4, 9]], dtype=torch.long)
    with torch.inference_mode():
        pytorch_logits = reference_model(input_ids).numpy()
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(["logits"], {"input_ids": input_ids.numpy()})[0]

    assert output_path.is_file()
    assert tokenizer_output_path.read_text() == "{}"
    assert (onnx_logits.argmax(axis=-1) == pytorch_logits.argmax(axis=-1)).all()
