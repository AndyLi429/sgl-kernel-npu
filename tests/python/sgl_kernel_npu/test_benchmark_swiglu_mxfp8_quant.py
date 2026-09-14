import importlib.util
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).with_name("benchmark_swiglu_mxfp8_quant.py")


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_swiglu_mxfp8_quant", _SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_rejects_hidden_size_without_complete_mx_groups():
    """Catch a benchmark that would compare incompatible MXFP8 output layouts."""
    benchmark = _load_benchmark_module()

    with pytest.raises(SystemExit):
        benchmark.parse_args(["--hidden-size", "3074"])
