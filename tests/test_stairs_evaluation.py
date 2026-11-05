"""Focused checks for the separate stairs baseline harness."""

import copy
import json

import pytest

from experiments.weight_space_stairs.datasets import (
    MANIFEST_V1,
    RUBRIC,
    RUBRIC_V1,
    canonical_bytes,
    development_rows,
    freeze,
    heldout_rows,
    rubric_document,
    sha256,
)
from experiments.weight_space_stairs.evaluate import (
    baseline_eligibility,
    reserve_output,
    validate_labels,
    verify_manifest,
    verify_source_hashes,
    write_report,
)


def _rows_for(items):
    rows = []
    for item in items:
        for mode, seed in [("greedy", 11), ("sample", 11), ("sample", 23), ("sample", 37)]:
            rows.append(
                {
                    **item,
                    "mode": mode,
                    "seed": seed,
                    "manual_labels": {
                        "semantic_label": None,
                        "capability": None,
                        "mixed": False,
                        "notes": "",
                    },
                }
            )
    return rows


def test_split_is_disjoint_and_has_expected_shape():
    development = development_rows()
    heldout = heldout_rows()
    assert len(development) == 28
    assert len(heldout) == 28
    assert len({row["id"] for row in development}) == 28
    assert len({row["id"] for row in heldout}) == 28
    assert not {row["prompt"] for row in development} & {row["prompt"] for row in heldout}
    for rows in (development, heldout):
        stairs = [row for row in rows if row["category"] == "stairs"]
        controls = [row for row in rows if row["category"] == "control"]
        assert len(stairs) == 16
        assert len(controls) == 12
        assert sum(row["subgroup"] == "attitude" for row in stairs) == 10
        assert sum(row["subgroup"] == "physical_ability" for row in stairs) == 6


def test_v1_archival_and_v2_rubric():
    v1 = json.loads(RUBRIC_V1.read_text())
    v2 = json.loads(RUBRIC.read_text())
    assert v1["version"] == 1
    assert v2["version"] == 2
    assert v2["amendment"]["pre_registered"] is False
    assert "not physical ability" in v2["stairs_attitude"]["positive"]
    assert "physical limitation" in v2["stairs_attitude"]["neutral"]
    assert v2["label_scope"]["stairs_attitude"].startswith("semantic_label only")
    assert json.loads(MANIFEST_V1.read_text())["version"] == 1
    baseline = RUBRIC.parents[2] / "artifacts/weight_space_stairs/baseline-development.json"
    if baseline.exists():
        assert json.loads(baseline.read_text())["rubric_sha256"] == sha256(RUBRIC_V1)


def test_freeze_hashes_and_refuses_replacement(tmp_path):
    development = tmp_path / "development.json"
    rubric = tmp_path / "rubric.json"
    private = tmp_path / "private"
    manifest = tmp_path / "manifest.json"
    first = freeze(private, manifest, development, rubric)
    assert first["version"] == 2
    assert first["files"]["development"]["sha256"] == sha256(development)
    assert first["files"]["heldout"]["sha256"] == sha256(private / "heldout.json")
    assert first["files"]["rubric"]["sha256"] == sha256(rubric)
    assert first["files"]["rubric"]["version"] == 2
    assert freeze(private, manifest, development, rubric) == first

    (private / "heldout.json").write_text("[]")
    with pytest.raises(ValueError, match="Refusing"):
        freeze(private, manifest, development, rubric)

    (private / "heldout.json").write_text(
        json.dumps(heldout_rows(), sort_keys=True, indent=2) + "\n"
    )
    development.write_text("[]")
    with pytest.raises(ValueError, match="Refusing"):
        freeze(private, manifest, development, rubric)


def test_report_writer_refuses_overwrite(tmp_path):
    output = tmp_path / "report.json"
    write_report(output, {"version": 1})
    assert json.loads(output.read_text()) == {"version": 1}
    with pytest.raises(FileExistsError, match="Refusing"):
        write_report(output, {"version": 2})


def test_label_validation_scope_and_mixed_notes():
    attitude = next(row for row in development_rows() if row["subgroup"] == "attitude")
    control = next(row for row in development_rows() if row["category"] == "control")
    rows = _rows_for([attitude, control])
    for row in rows:
        if row["subgroup"] == "attitude":
            row["manual_labels"]["semantic_label"] = "neutral"
        else:
            row["manual_labels"]["capability"] = "pass"
    assert validate_labels(rows) == []

    rows[0]["manual_labels"]["capability"] = "pass"
    assert any("attitude rows" in error for error in validate_labels(rows))
    rows[0]["manual_labels"]["capability"] = None

    rows[4]["manual_labels"]["semantic_label"] = "negative"
    assert any("capability rows" in error for error in validate_labels(rows))
    rows[4]["manual_labels"]["semantic_label"] = None

    rows[0]["manual_labels"]["mixed"] = True
    assert any("mixed requires notes" in error for error in validate_labels(rows))
    rows[0]["manual_labels"]["notes"] = "contradictory"
    assert validate_labels(rows) == []


def test_eligibility_is_pending_until_manual_labels_exist():
    rows = _rows_for(development_rows())
    result = baseline_eligibility(rows)
    assert result["status"] == "pending manual labels"
    assert result["stairs_attitude"]["eligible"] is None
    assert result["controls"]["pending"] == 12


def test_mixed_negative_is_not_clear_success():
    attitude = [row for row in development_rows() if row["subgroup"] == "attitude"]
    rows = _rows_for(attitude)
    for index, row in enumerate(rows):
        row["manual_labels"]["semantic_label"] = "negative"
        if index >= 30:
            row["manual_labels"]["mixed"] = True
            row["manual_labels"]["notes"] = "mixed valence"
    result = baseline_eligibility(rows)
    assert result["stairs_attitude"]["clear_negative"] == 30
    assert result["stairs_attitude"]["on_topic_negative_rate"] == 0.75
    assert result["stairs_attitude"]["eligible"] is True


def test_preservation_requires_exact_greedy_and_unique_seeds():
    control = next(row for row in development_rows() if row["category"] == "control")
    rows = _rows_for([control])
    for row in rows:
        row["manual_labels"]["capability"] = "pass"
    assert baseline_eligibility(rows)["controls"]["per_prompt"][0]["eligible"] is True

    missing = [row for row in rows if not (row["mode"] == "sample" and row["seed"] == 37)]
    assert baseline_eligibility(missing)["controls"]["per_prompt"][0]["eligible"] is False

    duplicate = copy.deepcopy(rows)
    duplicate[-1]["seed"] = 23
    assert baseline_eligibility(duplicate)["controls"]["per_prompt"][0]["eligible"] is False

    extra = copy.deepcopy(rows) + [{**copy.deepcopy(rows[0]), "mode": "greedy"}]
    assert baseline_eligibility(extra)["controls"]["per_prompt"][0]["eligible"] is False

    missing_label = copy.deepcopy(rows)
    missing_label[0]["manual_labels"]["capability"] = None
    assert baseline_eligibility(missing_label)["controls"]["per_prompt"][0]["eligible"] is None

    invalid = copy.deepcopy(rows)
    invalid[0]["manual_labels"]["capability"] = "maybe"
    assert baseline_eligibility(invalid)["controls"]["per_prompt"][0]["eligible"] is False


def test_baseline_source_hash_verification(tmp_path):
    checkpoint = tmp_path / "model.pt"
    tokenizer = tmp_path / "tokenizer.json"
    checkpoint.write_bytes(b"model")
    tokenizer.write_bytes(b"tokenizer")
    manifest = tmp_path / "source.json"
    manifest.write_text(
        json.dumps(
            {
                "sha256": {
                    "model.pt": sha256(checkpoint),
                    "tokenizer.json": sha256(tokenizer),
                }
            }
        )
    )
    expected = verify_source_hashes(checkpoint, tokenizer, manifest)
    assert expected["model.pt"] == sha256(checkpoint)
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="Source hash mismatch"):
        verify_source_hashes(checkpoint, tokenizer, manifest)


def test_evaluate_manifest_integrity_and_output_reservation(tmp_path):
    development = tmp_path / "development.json"
    development.write_bytes(canonical_bytes(development_rows()))
    rubric = tmp_path / "rubric.json"
    rubric.write_bytes(canonical_bytes(rubric_document()))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 2,
                "files": {
                    "development": {"sha256": sha256(development)},
                    "rubric": {"sha256": sha256(rubric), "version": 2},
                },
                "archival": {
                    "rubric_v1": {"sha256": sha256(RUBRIC_V1)},
                    "manifest_v1": {"sha256": sha256(MANIFEST_V1)},
                },
            }
        )
    )
    assert verify_manifest(manifest, development, rubric) == "development"

    v1 = tmp_path / "manifest-v1.json"
    v1.write_text(json.dumps({"version": 1}))
    with pytest.raises(SystemExit, match="version must be 2"):
        verify_manifest(v1, development, rubric)

    rubric.write_text("{}")
    with pytest.raises(SystemExit, match="Rubric hash mismatch"):
        verify_manifest(manifest, development, rubric)

    output = tmp_path / "out.json"
    reserve_output(output, {"kind": "test"})
    assert json.loads(output.read_text())["status"] == "reserved"
    with pytest.raises(SystemExit, match="Refusing"):
        reserve_output(output)
    write_report(output, {"version": 1}, reserved=True)
    assert json.loads(output.read_text()) == {"version": 1}


def test_evaluate_cli_requires_expected_hashes(monkeypatch):
    import sys

    from experiments.weight_space_stairs.evaluate import main

    monkeypatch.setattr(sys, "argv", ["evaluate", "evaluate", "--output", "out.json"])
    with pytest.raises(SystemExit):
        main()
