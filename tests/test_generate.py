from tiny_gpt.generate import parse_args


def test_generation_cli_parses_profile_without_loading_checkpoint():
    args = parse_args(["--profile", "smollm2_360m", "--prompt", "Hello"])

    assert args.profile == "smollm2_360m"
    assert args.prompt == "Hello"
