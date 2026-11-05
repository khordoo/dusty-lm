import json

from experiments.weight_space_traits import catalog


def test_tracked_manifest_matches_frozen_catalog():
    assert json.loads(catalog.MANIFEST.read_text()) == catalog.catalog_manifest()
    assert catalog.assert_frozen()["catalog_sha256"] == catalog.catalog_sha256()


def test_stage_splits_are_disjoint_and_do_not_reuse_prior_screens():
    report = catalog.disjointness_report()
    assert report["prior_prompt_overlap"] == []
    assert report["stage1_vs_stage2_overlap"] == []
    assert report["development_vs_heldout_overlap"] == []
    assert report["prior_screens"]["stairs_development"]["present"] is True
    assert report["prior_screens"]["cleaning_development"]["present"] is True


def test_public_stage1_catalog_hides_stage2_and_heldout_prompts():
    public = json.dumps(catalog.public_catalog())
    for item in catalog.TRAITS:
        for prompt in item["development"] + item["heldout"]:
            assert prompt not in public


def test_rubric_freezes_semantic_gate_and_rejects_keyword_success():
    assert catalog.GATE == 0.75
    assert catalog.RUBRIC["keyword_rule"].startswith("String or keyword occurrence is not")
    assert catalog.RUBRIC["gate"]["mixed_counts_as_positive"] is False
    assert catalog.RUBRIC["gate"]["exclusions_allowed"] is False
    assert catalog.RUBRIC["automatic_semantic_success"] is False
