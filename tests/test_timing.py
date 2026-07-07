import pytest

from dustylm.timing import format_duration, timed_step


def test_format_duration_uses_seconds_for_short_runs():
    assert format_duration(3.24) == "3.2s"


def test_format_duration_uses_minutes_for_longer_runs():
    assert format_duration(102) == "1m 42s"


def test_format_duration_uses_hours_for_long_runs():
    assert format_duration(3723) == "1h 2m 3s"


def test_timed_step_prints_stage_and_elapsed_time(monkeypatch, capsys):
    times = iter([10.0, 12.4])
    monkeypatch.setattr("dustylm.timing.perf_counter", lambda: next(times))

    with timed_step("Prepare tokenized datasets", stage="3/7"):
        print("working")

    output = capsys.readouterr().out
    assert "[3/7] Prepare tokenized datasets" in output
    assert "working" in output
    assert "✔ Completed in 2.4s" in output


def test_timed_step_reports_failure_before_reraising(monkeypatch, capsys):
    times = iter([1.0, 62.0])
    monkeypatch.setattr("dustylm.timing.perf_counter", lambda: next(times))

    with pytest.raises(RuntimeError, match="boom"):
        with timed_step("Train model"):
            raise RuntimeError("boom")

    output = capsys.readouterr().out
    assert "Train model" in output
    assert "✖ Failed after 1m 1s" in output
