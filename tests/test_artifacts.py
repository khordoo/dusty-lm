from dustylm.artifacts import list_downloadable_profiles, parse_args


def test_artifacts_cli_lists_only_downloadable_profiles():
    assert list_downloadable_profiles() == [
        "sft_smollm2_135m",
        "smollm2_135m",
        "smollm2_360m",
    ]


def test_artifacts_cli_parses_download_without_network():
    args = parse_args(["download", "--profile", "smollm2_360m", "--convert"])

    assert args.command == "download"
    assert args.profile == "smollm2_360m"
    assert args.convert is True
    assert args.force is False
