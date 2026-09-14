#!/usr/bin/env python3
"""Benchmark MXFP8 SwiGLU quantization on Ascend NPU.

Both paths use BF16 input, E4M3FN payloads, and one E8M0 scale per 32 output
elements. The AscendC path intentionally has no group_index: this measures the
same dense [tokens, hidden_size] workload as the Triton path.
"""

import argparse
import time

MX_QUANT = 2
MX_GROUP_SIZE = 32


def _positive_int(value: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return integer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=_positive_int, default=4096)
    parser.add_argument("--hidden-size", type=_positive_int, default=3072)
    parser.add_argument("--experts", type=_positive_int, default=64)
    parser.add_argument("--warmup", type=_positive_int, default=20)
    parser.add_argument("--iters", type=_positive_int, default=100)
    parser.add_argument("--clamp-value", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.hidden_size % (2 * MX_GROUP_SIZE) != 0:
        parser.error("--hidden-size must be divisible by 64 for MXFP8 SwiGLU")
    return args


def _make_group_list(torch, tokens, experts):
    counts = torch.full((experts,), tokens // experts, dtype=torch.int64, device="npu")
    counts[: tokens % experts] += 1
    return counts


def _benchmark(torch, fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1e6 / iterations


def main(argv=None):
    args = parse_args(argv)
    import torch
    import torch_npu  # noqa: F401
    from sgl_kernel_npu.activation.swiglu_mxfp8_quant import swiglu_quant

    if not hasattr(torch.ops.npu, "swiglu_group_quant"):
        raise RuntimeError(
            "torch.ops.npu.swiglu_group_quant is unavailable; rebuild and install "
            "an A5-enabled sgl_kernel_npu wheel."
        )

    torch.manual_seed(args.seed)
    x = torch.randn(args.tokens, args.hidden_size, dtype=torch.bfloat16, device="npu")
    group_list = _make_group_list(torch, args.tokens, args.experts)

    def triton_op():
        return swiglu_quant(
            x,
            group_list,
            group_list_type=1,
            need_quant=True,
            do_limit=args.clamp_value != 0.0,
            limit=args.clamp_value,
        )

    def ascendc_op():
        return torch.ops.npu.swiglu_group_quant(
            x,
            None,
            None,
            torch.float8_e4m3fn,
            MX_QUANT,
            MX_GROUP_SIZE,
            True,
            True,
            False,
            0,
            args.clamp_value,
        )

    triton_us = _benchmark(torch, triton_op, args.warmup, args.iters)
    ascendc_us = _benchmark(torch, ascendc_op, args.warmup, args.iters)
    print(f"device: {torch.npu.get_device_name(torch.npu.current_device())}")
    print(f"torch: {torch.__version__}; torch_npu: {torch_npu.__version__}")
    print(
        f"workload: tokens={args.tokens}, hidden_size={args.hidden_size}, "
        f"dtype=bf16, output=e4m3fn, scale=e8m0, group_size={MX_GROUP_SIZE}"
    )
    print(f"timing: warmup={args.warmup}, iters={args.iters}")
    print(
        f"triton_swiglu_quant: {triton_us:.2f} us, {args.tokens * 1e6 / triton_us:.2f} tokens/s"
    )
    print(
        f"ascendc_swiglu_group_quant: {ascendc_us:.2f} us, {args.tokens * 1e6 / ascendc_us:.2f} tokens/s"
    )
    print(f"ascendc_vs_triton: {triton_us / ascendc_us:.3f}x")


if __name__ == "__main__":
    main()
