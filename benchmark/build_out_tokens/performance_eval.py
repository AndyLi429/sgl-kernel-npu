"""Collect warmed BuildOutTokens kernels with the CANN msprof CLI.

Run from a configured container with the source package on PYTHONPATH:
  python benchmark/build_out_tokens/performance_eval.py --output /tmp/build_out_profile
Each implementation/shape gets a separate profile. Raw CSVs are retained.
"""

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def worker(args):
    import torch
    import torch_npu  # noqa: F401

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[2] / "tests/python/sgl_kernel_npu")
    )
    from sgl_kernel_npu.sample import build_out_tokens
    from test_build_out_tokens import make_inputs, torch_reference

    inputs = make_inputs(args.bs, args.gamma)
    # Baseline exactly matches SGLang: token/bonus int64, correct_len int32.
    fn = build_out_tokens if args.impl == "triton" else torch_reference
    for _ in range(20):
        fn(**inputs)
    torch.npu.synchronize()
    for _ in range(args.iterations):
        result = fn(**inputs)
    torch.npu.synchronize()
    assert result.shape == (args.bs, args.gamma + 1)


def summarize(directory, iterations):
    summaries = list(directory.rglob("op_summary*.csv"))
    if len(summaries) != 1:
        raise RuntimeError(
            f"Expected one op_summary CSV under {directory}: {summaries}"
        )
    with summaries[0].open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        duration = next(k for k in fields if k.replace(" ", "") == "TaskDuration(us)")
        groups = defaultdict(list)
        for row in reader:
            groups[row["Op Type"]].append(float(row[duration]))
    # Each profile contains 20 warmups followed by the measured calls. A native
    # operation may launch multiple kernels of the same type per function call.
    measured = {
        k: v[-(len(v) // (iterations + 20)) * iterations :]
        for k, v in groups.items()
        if len(v) >= iterations + 20
    }
    if not measured:
        raise RuntimeError(f"No repeated steady-state kernels in {summaries[0]}")
    return {
        "device_us": sum(sum(v) / iterations for v in measured.values()),
        "kernels": {k: sum(v) / iterations for k, v in measured.items()},
        "kernels_per_call": {k: len(v) // iterations for k, v in measured.items()},
        "csv": str(summaries[0]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--impl", choices=("torch", "triton"))
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--gamma", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("build_out_profile"))
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for bs in (1, 32, 128, 512):
        for gamma in (3, 7, 15):
            case = {"bs": bs, "gamma": gamma}
            for impl in ("torch", "triton"):
                directory = (args.output / f"bs{bs}_g{gamma}_{impl}").resolve()
                application = (
                    f"{sys.executable} {Path(__file__).resolve()} --worker "
                    f"--impl {impl} --bs {bs} --gamma {gamma} "
                    f"--iterations {args.iterations}"
                )
                subprocess.run(
                    ["msprof", f"--application={application}", f"--output={directory}"],
                    check=True,
                )
                case[impl] = summarize(directory, args.iterations)
            case["ratio"] = case["torch"]["device_us"] / case["triton"]["device_us"]
            results.append(case)
            (args.output / "results.json").write_text(json.dumps(results, indent=2))
            print(json.dumps(case), flush=True)


if __name__ == "__main__":
    main()
