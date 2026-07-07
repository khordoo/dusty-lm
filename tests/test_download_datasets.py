from data_pipeline.download_datasets import DEFAULT_TINYSTORIES_SLICE, parse_args


def test_download_datasets_defaults_to_100k_tinystories_slice():
    args = parse_args([])

    assert DEFAULT_TINYSTORIES_SLICE == "train[:100000]"
    assert args.tinystories_slice == "train[:100000]"
