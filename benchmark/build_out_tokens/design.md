# BuildOutTokens (DSpark)

The source operation is `BuildOutTokens` in SGLang's
`python/sglang/kernels/ops/speculative/dspark/dspark_verify_window.py`.
SGLang reference revision: `3c84568f2a9cc5c9625ff8f0b859a7496dc58427`.
Kernel repository baseline: `e280ae49b4329dba5a85bf807264bf011d17acd6`.

## Contract

Keyword-only interface: `build_out_tokens(draft_tokens, correct_len, bonus,
verify_num_draft_tokens, gamma)`, exported from `sgl_kernel_npu.sample`.
Inputs are on the same NPU: draft `[batch, gamma]`, lengths `[batch]`, and
bonus `[batch]`. Token and length inputs accept int32/int64; the result is
int64 `[batch, gamma + 1]`. `verify_num_draft_tokens == gamma + 1`.
The caller guarantees `0 <= correct_len <= gamma`.

For every request b and output position k:

```text
out[b,k] = bonus[b]       if k == correct_len[b]
           draft[b,k]    otherwise if k < gamma
           0             otherwise
```

The suffix after the accepted prefix is retained, matching the full Torch
buffer. The operation does not calculate acceptance lengths or trim output.
Empty batches return an empty tensor without launching a kernel. Strided
inputs are read directly, without conversion or contiguous copies.

## Implementation and review

One Triton kernel fuses copy, zero fill, dtype widening and bonus insertion.
The 1D launch uses the detected vector-core count. Each core handles 256
output elements at a time, then advances by the physical grid width.
All loads and stores are masked. Output indices have one writer each.
There are no atomics, reductions, host reads of tensor values, or FP casts.
The live tile consists of only a few KB of values and indices, below UB capacity.

CANN 9.1's compiler failed on `tl.where` selecting int64 values with an
i1/i8 mask mismatch. Bitwise selection with an all-zero/all-one int64 mask
preserves every token bit and compiles in the target container.

Static review: no unresolved P0/P1 findings. Performance must be established
by the retained msprof CSVs, not inferred from kernel fusion alone.

## Validation

Run with this checkout's package installed or on its package path:

```bash
export ASCEND_RT_VISIBLE_DEVICES=15
python benchmark/build_out_tokens/precision_eval.py
python benchmark/build_out_tokens/performance_eval.py --output /tmp/build_out_profile
```

Precision uses integer exact equality against Torch's copy/fill/scatter result.
The matrix includes empty/single/nonaligned/large batches, zero and nonzero
gamma, int32/int64 lengths and tokens, strided inputs, negative tokens and
int64 token values outside int32 range. Separate tests force zero/full acceptance.

Performance uses identical NPU inputs for both implementations, 20 warmups
and 100 steady-state calls per case. `Task Duration(us)` comes from msprof;
the last 100 calls' device kernel durations are summed per function call.
The default matrix is batch 1/32/128/512 and gamma 3/7/15. The result is
operator device time, not server TPOT or throughput. No SGLang dispatch
integration is included in this kernel-library change.
