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

## Review fix round 1 (BLOCKED)

Two review-required fixes are implemented and locally verified:

- Added `__schedmode__(1)` to the direct-launch Compressor kernel entry. This
  is the documented Kernel-direct equivalent of upstream
  `context->SetScheduleMode(BATCH_MODE_SCHEDULE)`. It changes launch scheduling
  metadata only; kernel math, VF code, and synchronization calls are unchanged.
  The CANN Kernel-direct documentation specifically requires the
  `__schedmode__(mode)` qualifier and recommends mode 1 for kernels with
  cross-core synchronization in multi-stream execution:
  <https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10053.html>.
- Captured the `aclrtMemcpy` return value and required `ACL_SUCCESS` before the
  kernel launch. A failed tiling H2D copy therefore cannot fall through into a
  launch with invalid tiling memory.

The focused source-contract test was written before these changes. Its RED
result was:

```text
test_direct_launch_uses_batch_schedule_mode ... FAIL
test_tiling_copy_is_checked_before_kernel_launch ... ERROR
Ran 2 tests
FAILED (failures=1, errors=1)
```

After the two fixes, its GREEN result was:

```text
python -m unittest tests.python.sgl_kernel_npu.test_compressor_source_contract -v
test_direct_launch_uses_batch_schedule_mode ... ok
test_tiling_copy_is_checked_before_kernel_launch ... ok
Ran 2 tests in 0.001s
OK
```

### Exact dispatcher blocker

Task 3 remains **BLOCKED** on template-kernel dispatch. The current wrapper
computes a six-field template TilingKey, but the checked-in generic
`EXEC_KERNEL_CMD` expands to:

```text
ACLRT_LAUNCH_KERNEL(kernel_name)(blockdim, acl_stream, params...)
```

It has no TilingKey, template-parameter, binary-handle, or function-handle
argument. Therefore it cannot prove or request selection among the
`compressor<XLayout, XDType, Coff, CacheMode, TemplateId, GradEnabled>` kernel
instances. Passing the computed key to this generic macro would require
inventing an ABI and is intentionally not done.

The exact missing build product is the A5 CANN-generated
`aclrtlaunch_compressor.h`, together with the matching compiled
`workspace_kernel` launcher/object metadata that maps the Compressor template
TilingKey to a concrete kernel instance. The source-only workstation has none
of the following required evidence:

- the declaration and argument order of `aclrtlaunch_compressor`;
- the generated Compressor instance symbols/object metadata;
- a generated dispatcher proving whether and how the TilingKey reaches those
  instances.

The upstream framework route does carry the key separately from `blockDim`:

- `D:/GitCode/ops-transformer/attention/compressor/op_host/arch35/compressor_tiling.cpp`
  computes `GET_TPL_TILING_KEY(...)`, then calls
  `context->SetTilingKey(compressorContext.tilingKey)` and
  `context->SetBlockDim(compressorContext.blockDim)`.
- `D:/GitCode/ops-transformer/tests/ut/framework_special/stubs/runtime/runtime_stubs.cpp`
  declares the corresponding runtime ABI as
  `rtKernelLaunchWithHandleV2(void *hdl, const uint64_t tilingKey,
  uint32_t blockDim, ...)` and also declares
  `rtBinaryGetFunction(binHandle, tilingKey, funcHandle)`.

This is evidence for the upstream binary-handle/template-key route, not
authorization to call those runtime APIs from this repository: the direct
static-library wrapper has neither the required binary handle nor a verified
generated argument pack/config contract.

Run this exact build and inspection on an A5 development host with the matching
CANN toolkit and `torch-npu` installed:

```bash
ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest \
  bash build.sh -a kernels Ascend950PR_9599

launcher_header="$(find build -type f -name aclrtlaunch_compressor.h -print -quit)"
test -n "$launcher_header"
sed -n '1,220p' "$launcher_header"

find build -type f \
  \( -name '*compressor*.o' -o -name '*compressor*.json' \
     -o -name 'libworkspace_kernel.a' \) -print

workspace_archive="$(find build -type f -name libworkspace_kernel.a -print -quit)"
test -n "$workspace_archive"
nm -A -C "$workspace_archive" | grep -i compressor
```

The generated header and symbol/metadata output must be inspected before
changing the host call. If they do not expose a TilingKey-aware direct
dispatcher, the build integration must use the upstream framework binary
launcher route (or a CANN-supported generated equivalent); a generic direct
launch is not sufficient. No Task 4 cache/capture work should proceed until
this dispatch contract is resolved.

The fresh local A5 build attempt confirms that those products cannot be
generated in this environment:

```text
bash build.sh -a kernels Ascend950PR_9599
Build target: kernels
CMake SOC_VERSION: Ascend950PR_9599
Error: Cannot find an Ascend toolkit directory containing set_env.sh
```

Files added or changed by review fix round 1:

- `csrc/compressor/op_host/compressor.cpp`
- `csrc/compressor/op_kernel/compressor.cpp`
- `tests/python/sgl_kernel_npu/test_compressor_source_contract.py`
- `.superpowers/sdd/2026-08-19-a5-compressor-operator-migration/task-3-report.md`
