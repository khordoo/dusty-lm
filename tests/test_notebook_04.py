import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/04_hf_export_and_web_ui.ipynb")


def load_notebook():
    return json.loads(NOTEBOOK_PATH.read_text())


def test_notebook_04_has_stable_unique_cell_ids_and_no_saved_outputs():
    notebook = load_notebook()
    cell_ids = [cell.get("id") for cell in notebook["cells"]]

    assert notebook["nbformat"] == 4
    assert all(cell_ids)
    assert len(cell_ids) == len(set(cell_ids))
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_notebook_04_cannot_publish_or_start_server_during_run_all():
    notebook = load_notebook()
    active_lines = [
        line.strip()
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        for line in cell["source"]
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert not any("make push-hub" in line for line in active_lines)
    assert not any("make push-dataset" in line for line in active_lines)
    assert not any("make serve-web" in line for line in active_lines)


def test_notebook_04_uses_current_hugging_face_commands():
    source = NOTEBOOK_PATH.read_text()

    assert "hf auth login" in source
    assert "You do not need both" in source
    assert "huggingface-cli" not in source
    assert "HUB_TARGET=dataset" not in source


def test_notebook_04_requires_learner_owned_destination_repositories():
    notebook = load_notebook()
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert 'HF_MODEL_REPO_ID = "your-username/dusty-8m-sft"' in source
    assert 'HF_DATASET_REPO_ID = "your-username/dusty-chat"' in source
    assert "`HF_MODEL_REPO_ID` is the exact upload destination" in source
    assert "`HF_DATASET_REPO_ID`; that variable is the exact upload destination" in source
