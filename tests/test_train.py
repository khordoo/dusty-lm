from tiny_gpt.train import parse_args


def test_train_parse_args_defaults_to_one_epoch():
    args = parse_args([])

    assert args.epochs == 1


def test_train_parse_args_accepts_epoch_override():
    args = parse_args(["--epochs", "3"])

    assert args.epochs == 3
