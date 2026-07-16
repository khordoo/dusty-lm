from data_pipeline.generate_sft import build_prompt


def test_build_prompt_describes_dustys_world_and_requested_category():
    messages = build_prompt("crumbs", n_examples=5)

    assert messages[0]["role"] == "system"
    assert "Dusty lives in a small house." in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "Generate 5 examples" in messages[1]["content"]
    assert "Category: crumbs" in messages[1]["content"]
