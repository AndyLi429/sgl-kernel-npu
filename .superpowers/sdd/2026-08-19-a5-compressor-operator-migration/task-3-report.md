# Task 3 Report: Compressor host validation and tiling adaptation

## Implementation

- Added the A5-only PyTorch host wrapper at
  `csrc/compressor/op_host/compressor.cpp` and listed it in `OP_SRCS` under
  `SGL_KERNEL_ENABLE_A5_ONLY_OPS`.
- Moved the Compressor kernel include directory under the same A5 gate. This
  completes the deferred Task 2 cleanup and leaves the non-A5 workspace target
  without a Compressor include dependency.
- Added metadata-only validation for required and optional tensors. Validation
  covers NPU device consistency, strided/contiguous layout where required,
  dtype, rank, BSH/TH layout, supported sizes, `cmp_ratio`, `coff`,
  `cache_mode`, and the Linear/Ring `state_block_table` rank. It does not read
  tensor values or use `.item()`, boolean tensor evaluation, or scalar
  extraction.
- Derived the internal state stride from `state_cache.stride(0)` as an
  `int64_t`; no public stride argument was added.
- Added inference-only native ABI outputs with `requires_grad(false)` and the
  upstream inferred shapes/dtypes. The public function still returns only
  `cmp_kv`, while `state_cache` remains the mutable public input.
- Adapted the imported A5 tiling to the repository's `ge_helper::TilingContext`
  without changing its scheduling calculations, template decisions,
  workspace calculations, device math, or kernel synchronization. Optional
  input absence is translated from the helper's null tensor representation to
  the native tiling's null descriptor/shape representation.
- Added only the two helper capabilities proven necessary by the imported
  tiling: `AttrDef::Bool` and deterministic-level set/get.
- Mapped `at::globalContext().deterministicAlgorithms()` to deterministic
  level `3` when enabled and `0` when disabled. The level is copied into the
  upstream `batchConsistency` tiling field.
- Forced the private `grad_enabled` attribute to `false` and corrected the
  imported tiling-key conversion to dereference that boolean value rather than
  treating the non-null attribute pointer as true.
- Added the uncached eager tiling upload and direct launch used as Task 4's
  cache/capture baseline. The launch retains the tiling-selected full AIC
  `blockDim`; no device kernel source was changed.

## TDD evidence

The invalid-host-metadata test was added before the wrapper. The prescribed
test could not collect locally because the package/`torch_npu` runtime is not
installed. A dependency-free source check recorded the expected RED state:

```text
RED: compressor host wrapper is absent
```

After adding metadata validation, the source audit passed:

```text
GREEN(static): required metadata validation is present and forbidden value reads are absent
```

The deterministic forwarding assertion was then added and failed before its
implementation:

```text
RED: deterministic mode is not forwarded to tiling
```

After adding the 3/0 mapping and context forwarding, the source audit passed:

```text
GREEN(static): deterministic mode is mapped to level 3/0 and forwarded to tiling
```

The Python test uses `subTest` parameterization for the exact required cases:

- Ring mode rejects a 2D `state_block_table`.
- Linear mode rejects a 1D `state_block_table`.
- Both modes reject a non-FP32 `state_cache` before launch.

## Verification

Passed:

```text
python -m py_compile tests/python/sgl_kernel_npu/test_compressor.py
git diff --check
pre-commit run --files csrc/CMakeLists.txt csrc/compressor/op_host/compressor.cpp csrc/compressor/op_host/tiling/compressor_tiling.cpp csrc/compressor/op_host/tiling/compressor_tiling.h csrc/utils/ge_helper.h tests/python/sgl_kernel_npu/test_compressor.py
```

The final pre-commit run passed all applicable hooks, including Python AST,
isort, formatting, codespell, and clang-format.

Blocked by the local environment:

```text
PYTHONPATH=python/sgl_kernel_npu python tests/python/sgl_kernel_npu/test_compressor.py -k invalid_host_metadata
PYTHONPATH=python/sgl_kernel_npu python tests/python/sgl_kernel_npu/test_compressor.py -k deterministic
```

Both stop during package import with:

```text
ModuleNotFoundError: No module named 'torch_npu'
```

```text
bash build.sh -a kernels
```

Stops before CMake/configuration with:

```text
Error: Cannot find an Ascend toolkit directory containing set_env.sh
```

No A5+CANN compilation, generated-launcher validation, installed-wheel test,
or real-device execution is claimed.

## Files changed

- `csrc/CMakeLists.txt`
- `csrc/compressor/op_host/compressor.cpp`
- `csrc/compressor/op_host/tiling/compressor_tiling.cpp`
- `csrc/compressor/op_host/tiling/compressor_tiling.h`
- `csrc/utils/ge_helper.h`
- `tests/python/sgl_kernel_npu/test_compressor.py`
- `.superpowers/sdd/2026-08-19-a5-compressor-operator-migration/task-3-report.md`

## Concerns and next verification

- Rebuild and reinstall the A5 wheel on an Ascend950 machine with CANN and
  `torch-npu`, then run the focused invalid-metadata and deterministic tests.
- Task 4 must replace the uncached per-call tiling allocation/upload with its
  mutex-protected, per-device, capture-safe cache. Its key must include the
  deterministic level already computed here so deterministic-on/off calls
  cannot share an entry.
- Verify the generated `aclrtlaunch_compressor.h` direct-launch ABI during the
  A5 build. That generated header is unavailable on this workstation.
