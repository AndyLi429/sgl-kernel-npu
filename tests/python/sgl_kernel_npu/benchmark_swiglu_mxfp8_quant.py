"""Benchmark MXFP8 SwiGLU quantization on Ascend NPU.

Both paths use BF16 input, E4M3FN payloads, and one E8M0 scale per 32 output
elements. The AscendC path intentionally has no group_index: this measures the
same dense [tokens, hidden_size] workload as the Triton path.
"""

import time

MX_QUANT = 2
MX_GROUP_SIZE = 32
TOKENS = 4096
HIDDEN_SIZE = 3072
EXPERTS = 64
WARMUP = 20
ITERATIONS = 100
CLAMP_VALUE = 7.0
SEED = 0


def _make_group_list(torch, tokens, experts):
    counts = torch.full((experts,), tokens // experts, dtype=torch.int64, device="npu")
    counts[: tokens % experts] += 1
    return counts


def _benchmark(torch, name, fn, warmup, iterations):
    print(f"{name}: warmup ({warmup} iterations)...", flush=True)
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    print(f"{name}: timing ({iterations} iterations)...", flush=True)
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - start) * 1e6 / iterations


def main():
    print("loading torch and torch_npu...", flush=True)
    import torch
    import torch_npu  # noqa: F401
    from sgl_kernel_npu.activation.swiglu_mxfp8_quant import swiglu_quant

    if not hasattr(torch.ops.npu, "swiglu_group_quant"):
        raise RuntimeError(
            "torch.ops.npu.swiglu_group_quant is unavailable; rebuild and install "
            "an A5-enabled sgl_kernel_npu wheel."
        )

    print("preparing shared MXFP8 input...", flush=True)
    torch.manual_seed(SEED)
    x = torch.randn(TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16, device="npu")
    group_list = _make_group_list(torch, TOKENS, EXPERTS)

    def triton_op():
        return swiglu_quant(
            x,
            group_list,
            group_list_type=1,
            need_quant=True,
            do_limit=CLAMP_VALUE != 0.0,
            limit=CLAMP_VALUE,
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
            CLAMP_VALUE,
        )

    triton_us = _benchmark(torch, "triton_swiglu_quant", triton_op, WARMUP, ITERATIONS)
    ascendc_us = _benchmark(
        torch, "ascendc_swiglu_group_quant", ascendc_op, WARMUP, ITERATIONS
    )
    print(f"device: {torch.npu.get_device_name(torch.npu.current_device())}")
    print(f"torch: {torch.__version__}; torch_npu: {torch_npu.__version__}")
    print(
        f"workload: tokens={TOKENS}, hidden_size={HIDDEN_SIZE}, "
        f"dtype=bf16, output=e4m3fn, scale=e8m0, group_size={MX_GROUP_SIZE}"
    )
    print(f"timing: warmup={WARMUP}, iters={ITERATIONS}")
    print(
        f"triton_swiglu_quant: {triton_us:.2f} us, {TOKENS * 1e6 / triton_us:.2f} tokens/s"
    )
    print(
        f"ascendc_swiglu_group_quant: {ascendc_us:.2f} us, {TOKENS * 1e6 / ascendc_us:.2f} tokens/s"
    )
    print(f"ascendc_vs_triton: {triton_us / ascendc_us:.3f}x")


if __name__ == "__main__":
    main()
