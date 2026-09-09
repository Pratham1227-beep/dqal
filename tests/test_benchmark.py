"""
Integration test running a quick benchmark simulation to verify end-to-end correctness.
"""

import os
import pytest
from benchmarks.run_benchmark import run_full_benchmark


def test_full_benchmark_execution(tmp_path):
    plot_file = str(tmp_path / "test_benchmark.png")
    run_full_benchmark(output_plot_path=plot_file)
    assert os.path.exists(plot_file)
    assert os.path.getsize(plot_file) > 1000
