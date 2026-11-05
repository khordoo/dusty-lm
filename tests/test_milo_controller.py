import os

import pytest
import torch

from experiments.weight_space_milo import controller as c
from experiments.weight_space_milo.launch import assert_text_only, request_options
from experiments.weight_space_milo.tensor_worker import validate


@pytest.fixture
def run(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    torch.save(
        {"embed.weight": torch.zeros(2, 2), "layers.0.attention.proj.weight": torch.eye(2)},
        source / "model.pt",
    )
    (source / "tokenizer.json").write_text("{}")
    run = tmp_path / "run"
    run.mkdir()
    tensor = run / "tensor"
    tensor.mkdir(mode=0o777)
    tensor.chmod(0o777)
    import shutil

    shutil.copyfile(source / "model.pt", tensor / "current.pt")
    (tensor / "current.pt").chmod(0o666)
    c.write_json(run / "protocol.json", c.PROTOCOL)
    image = c.docker_image() if os.getenv("MILO_DOCKER") == "1" else "fixture-image"
    state = {
        "protocol": c.PROTOCOL,
        "protocol_sha256": c.sha256(run / "protocol.json"),
        "source": str(source),
        "image": image,
        "counts": dict.fromkeys(c.LIMITS, 0),
        "candidates": {},
        "final": None,
    }
    c.write_json(run / "state.json", state)
    monkeypatch.setattr(c, "verify", lambda path: {})
    return run


@pytest.mark.parametrize(
    "code",
    [
        "import os",
        'open("/etc/passwd")',
        "state.clear()",
        "torch.rand(2)",
        'torch.linalg.solve(state["x"],state["x"])',
        "for x in state: print(x)",
        "torch=state",
        "print.__globals__",
        "x = [a for a in state]",
    ],
)
def test_restricted_python_refusals(code):
    with pytest.raises(ValueError):
        validate(code)


def test_budgets_charge_rejections_and_survive_restart(run):
    controller = c.Controller(run)
    for _ in range(100):
        assert "error" in controller.dispatch({"action": "python", "code": "import os"})
    with pytest.raises(ValueError, match="budget exhausted"):
        c.Controller(run).dispatch({"action": "python", "code": "print(1)"})
    for _ in range(25):
        assert "error" in controller.dispatch({"action": "evaluate", "sequences": []})
    with pytest.raises(ValueError, match="budget exhausted"):
        controller.dispatch({"action": "evaluate", "sequences": []})


def test_candidate_budget_and_final_lock(run):
    controller = c.Controller(run)
    for i in range(10):
        assert controller.dispatch({"action": "submit"})["candidate"] == f"candidate-{i + 1:02d}"
    with pytest.raises(ValueError, match="budget exhausted"):
        controller.dispatch({"action": "submit"})
    assert (
        controller.dispatch({"action": "final", "candidate": "candidate-01"})["final"]
        == "candidate-01"
    )
    with pytest.raises(ValueError, match="finalized"):
        controller.dispatch({"action": "submit"})


def test_api_has_no_tools_and_refuses_unexpected_tool_event():
    from types import SimpleNamespace

    opts = request_options([])
    assert opts["tools"] == [] and opts["tool_choice"] == "none"
    assert opts["reasoning"] == {"effort": "high", "context": "all_turns"}
    with pytest.raises(ValueError):
        assert_text_only(
            SimpleNamespace(status="completed", output=[SimpleNamespace(type="function_call")])
        )


@pytest.mark.skipif(
    os.getenv("MILO_DOCKER") != "1", reason="Run make milo-verify for Linux Docker integration"
)
def test_real_docker_tensor_edit_frozen_rejection_and_scratch(run):
    controller = c.Controller(run)
    result = controller.dispatch(
        {
            "action": "python",
            "code": 'state["layers.0.attention.proj.weight"][0,0] = 2.0\nscratch["v"] = state["layers.0.attention.proj.weight"].clone()\nprint(scratch["v"])',
        }
    )
    assert "error" not in result, result
    assert result["audit"]["changed_entries"] == 1
    assert "error" not in controller.dispatch(
        {"action": "python", "code": 'print(scratch["v"][0,0])'}
    )
    before = (run / "tensor/current.pt").read_bytes()
    refused = controller.dispatch({"action": "python", "code": 'state["embed.weight"][0,0] = 9.0'})
    assert "Frozen tensor" in refused["error"]
    assert (run / "tensor/current.pt").read_bytes() == before
    restored = controller.dispatch(
        {
            "action": "python",
            "code": 'state["layers.0.attention.proj.weight"] = original["layers.0.attention.proj.weight"]',
        }
    )
    assert "error" not in restored, restored
    assert restored["audit"]["changed_entries"] == 0
