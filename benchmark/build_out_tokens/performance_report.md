# BuildOutTokens performance validation

Status: **not measured; performance improvement has not been established**.

## Baseline and candidate

- Baseline: SGLang's Torch-native int64 output allocation, draft `copy_`,
  last-column `fill_(0)`, int32-to-int64 acceptance-length conversion,
  and bonus `scatter_` on the same NPU.
- Candidate: one Triton vector kernel, reading input strides directly and
  retaining the complete int64 output buffer.
- Matrix: batch 1/32/128/512 and gamma 3/7/15.
- Each implementation: 20 warmups and 100 measured calls per case.
- Intended measurement: CANN `msprof` CLI, exported `Task Duration(us)`.
- Ratio: native device time / Triton device time; above 1 means faster.

The collector retains each implementation/shape's raw profile, excludes
setup-only kernels, and sums the final 100 calls' repeated kernel durations.
It accounts for multiple kernels of the same type within a function call.
The CSV aggregation was checked with synthetic data; actual profiler CSVs
and the collector still require validation on the target hardware.

## Results

| Metric | Result |
| --- | --- |
| Torch-native device duration | Not collected |
| Triton device duration | Not collected |
| Speedup ratio | Unknown |
| Memory bandwidth / vector utilization | Not collected |
| L2 hit rate / bank conflicts | Not collected |
| DSV4 TPOT / throughput impact | Not measured |

Remote hardware became unavailable during the full precision run, before
performance collection. No performance optimization was selected from timings.
The CANN 9.1 int64-selection compilation fix is documented in the precision
report. Fusion reduces the number of launches in the implementation, but that
alone is not evidence of lower measured latency.

## Reproduce after full precision passes

```bash
export ASCEND_RT_VISIBLE_DEVICES=15
python benchmark/build_out_tokens/performance_eval.py --output /tmp/build_out_profile
```

Inspect the CSV kernels and counts before accepting `results.json`.
Repeat measurements if there is contention. At the original validation start,
all NPUs had model memory allocations; reported compute utilization was 0%.
This was a shared device, not an exclusive benchmark environment.

The library API is exported as `sgl_kernel_npu.sample.build_out_tokens`.
SGLang's NPU `BuildOutTokens.execute` dispatch was not changed. Even a future
operator speedup will not affect serving until that dispatch uses this API.
