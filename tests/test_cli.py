"""
Unit tests for DQAL command-line interface (CLI).
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from dqal.cli import main


@pytest.fixture
def temp_datasets():
    rng = np.random.RandomState(42)
    clean_df = pd.DataFrame(
        rng.randn(100, 4),
        columns=["col_0", "col_1", "col_2", "col_3"],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = os.path.join(tmpdir, "train.csv")
        test_path = os.path.join(tmpdir, "test.csv")
        bad_path = os.path.join(tmpdir, "bad.csv")

        clean_df.to_csv(train_path, index=False)
        clean_df.iloc[:30].to_csv(test_path, index=False)

        bad_df = clean_df.copy()
        bad_df.iloc[:, :] = np.nan
        bad_df.to_csv(bad_path, index=False)

        yield {
            "train": train_path,
            "test": test_path,
            "bad": bad_path,
        }


def test_cli_version(capsys):
    ret = main(["version"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "dqal" in captured.out


def test_cli_missing_file():
    ret = main(["check", "does_not_exist.csv"])
    assert ret == 1


def test_cli_check_passed(temp_datasets, capsys):
    ret = main(["check", temp_datasets["test"], "--baseline", temp_datasets["train"]])
    assert ret == 0
    captured = capsys.readouterr()
    assert "PASSED" in captured.out
    assert "DQAL Data Quality Report" in captured.out


def test_cli_check_compact(temp_datasets, capsys):
    ret = main(["check", temp_datasets["test"], "--baseline", temp_datasets["train"], "--compact"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Data Quality Score (Q)" in captured.out


def test_cli_check_json(temp_datasets, capsys):
    import json
    ret = main(["check", temp_datasets["test"], "--baseline", temp_datasets["train"], "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["passed"] is True
    assert "quality_score" in parsed


def test_cli_check_blocked(temp_datasets, capsys):
    ret = main(["check", temp_datasets["bad"], "--baseline", temp_datasets["train"]])
    assert ret == 2
    captured = capsys.readouterr()
    assert "BLOCKED" in captured.out


def test_cli_convenience_invocation(temp_datasets, capsys):
    # Tests 'dqal <file>' shortcut
    ret = main([temp_datasets["test"]])
    assert ret == 0
    captured = capsys.readouterr()
    assert "DQAL Data Quality Report" in captured.out
