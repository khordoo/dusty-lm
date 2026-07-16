from types import SimpleNamespace

import pytest

from data_pipeline.download_datasets import (
    DEFAULT_TINYSTORIES_SLICE,
    parse_args,
    resolve_category_name,
)


def test_download_datasets_defaults_to_100k_tinystories_slice():
    args = parse_args([])

    assert DEFAULT_TINYSTORIES_SLICE == "train[:100000]"
    assert args.tinystories_slice == "train[:100000]"


def test_resolve_category_name_converts_class_label_index():
    feature = SimpleNamespace(names=["crumbs", "dusty_friends"])

    assert resolve_category_name(1, feature) == "dusty_friends"


def test_resolve_category_name_preserves_string_category():
    assert resolve_category_name("crumbs", SimpleNamespace(names=None)) == "crumbs"


def test_resolve_category_name_rejects_unknown_label():
    with pytest.raises(ValueError, match="Could not resolve Dusty category label"):
        resolve_category_name(4, SimpleNamespace(names=["crumbs"]))
