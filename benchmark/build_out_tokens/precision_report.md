# BuildOutTokens precision validation

Status: **partial hardware validation; full acceptance pending**.

## Environment

- Date: 2026-10-24.
- Host: `192.168.25.212`.
- Container: `yang-eplb-b103-d-20261023` (user corrected the initial container name).
- Device: physical device 15, compiler target `Ascend910_9362`.
- Device properties: 20 AI cores / 40 vector cores.
- CANN binary path: `/usr/local/Ascend/cann-9.1.0/bin`.
- torch: `2.10.0+cpu`; torch_npu: `2.10.0.post6`; Triton: `3.2.0`.
- Isolated source and logs: `/tmp/build_out_tokens_20261024` inside the container.
- Installed packages were not overwritten; an isolated `sitecustomize.py` loaded
  the new source module into the installed sample package for testing.

## Criterion

Integer output must match the entire Torch reference buffer exactly, including
the suffix beyond the accepted prefix. No floating-point tolerance applies.
MERE/MARE are not used for integer validation; exact equality is stronger.

## Observed results

| Check | Result |
| --- | --- |
| NPU zero/full acceptance, gamma 1/7/31, batch 33 | **6/6 passed** |
| NPU boundary unittest method | `Ran 1 test in 8.251s`, `OK`, exit 0 |
| Full NPU matrix: 560 shape/dtype/layout combinations | Started; final result unavailable |
| Invalid-width validation | Included in full suite; final result unavailable |
| Host reference versus bitwise formula, gamma 0/1/3/7/31 | 5/5 passed; CPU only |

The boundary test used int64 tokens (including values outside int32 range)
and int32 acceptance lengths. Its log is `smoke2.log` with `smoke2.exit=0`.
The full suite logs are `precision2.log` and `precision2.exit`. At the last
successful check the full suite was still running and had not reported a new
failure. SSH connectivity was then lost, and the user confirmed that remote
hardware was no longer available. **An unfinished test is not a pass.**

The intended matrix covers batch 0/1/2/7/31/32/33/127/512/2049;
gamma 0/1/3/7/8/15/31; int32/int64 tokens and acceptance lengths;
contiguous and strided inputs. Int32 bonus inputs are widened for the native
Torch scatter reference, which requires int64 source for an int64 destination.

## Implementation history

The initial int64 `tl.where` failed compilation in the target CANN compiler
with an i1/i8 mask type mismatch. Replacing selection with integer bitwise
masks fixed that compilation failure; the six boundary cases above passed
on the revised kernel. No precision tolerance or token narrowing was introduced.

## Remaining acceptance

When hardware is available, rerun:

```bash
export ASCEND_RT_VISIBLE_DEVICES=15
python benchmark/build_out_tokens/precision_eval.py
```

Only after the entire suite passes should the performance evaluation proceed.
Neither A5 validation nor full DSV4 inference accuracy was performed.
