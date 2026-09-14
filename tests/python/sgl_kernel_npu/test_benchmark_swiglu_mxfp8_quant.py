import importlib.util
from pathlib import Path

_SCRIPT_PATH = Path(__file__).with_name("benchmark_swiglu_mxfp8_quant.py")


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_swiglu_mxfp8_quant", _SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeNpu:
    def __init__(self):
        self.synchronize_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1


class _FakeTorch:
    def __init__(self):
        self.npu = _FakeNpu()


def test_benchmark_prints_progress_before_running_each_operator(capsys):
    """Catch a benchmark that appears stuck while the first operator is compiling."""
    benchmark = _load_benchmark_module()
    torch = _FakeTorch()
    calls = []

    benchmark._benchmark(
        torch, "triton", lambda: calls.append(None), warmup=1, iterations=2
    )

    assert calls == [None, None, None]
    assert torch.npu.synchronize_calls == 2
    assert "triton: warmup" in capsys.readouterr().out
