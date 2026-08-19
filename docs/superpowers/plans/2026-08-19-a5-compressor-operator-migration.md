# A5 Compressor Operator Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the ops-transformer A5 Compressor as an A5-only, inference-only `torch.ops.npu.compressor` operator that preserves the native Linear and Ring state semantics.

**Architecture:** Copy the A5/arch35 host tiling and device implementation from upstream without altering its maths, VF path, templates, or core synchronization. Add a small PyTorch host adapter that validates metadata without device-to-host synchronization, derives state stride from `state_cache.stride(0)`, creates private ABI-required outputs, caches graph-safe tiling per device, and launches the single kernel entry.

**Tech Stack:** PyTorch C++ extension, torch-npu/PrivateUse1 dispatch, CANN AscendC (`arch35`), ACL runtime direct launch, Python `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-19-a5-compressor-sglang-integration-design.md` (Phase A only; SGLang integration, prefill gate conclusion, and PD wire protocol are separately executable milestones.)

## Global Constraints

- Preserve the upstream baseline `D:/GitCode/ops-transformer/attention/compressor` at commit `5520571a4`, including its A5 mathematical logic, tiling structures, VF implementation, template selection, and full-core synchronization.
- Compile, declare, and register this operator only under `SGL_KERNEL_ENABLE_A5_ONLY_OPS`; do not change A3 behavior or registration.
- Public schema is inference-only and returns only `cmp_kv`; `state_cache` is mutable via `Tensor(a!)`.
- Always derive the internal state stride in elements from `state_cache.stride(0)`; never expose it as a public argument.
- Do not inspect NPU tensor values (`.item()`, boolean tensor evaluation, or equivalent) in host validation.
- Capture/replay may use only an existing per-device tiling cache entry. A cache miss during capture must fail before allocation or H2D tiling creation.
- Do not copy any current-worktree `kv_compress_epilog` changes. Existing untracked files are user-owned; do not stage or commit them.

---

## File structure

- `csrc/compressor/op_host/compressor.cpp`: PyTorch validation, tiling-context adaptation, per-device tiling cache, private outputs/workspace, and kernel launch.
- `csrc/compressor/op_host/tiling/compressor_tiling.{h,cpp}` and `compressor_tiling_data.h`: upstream A5 tiling contract adapted to the repository helper API.
- `csrc/compressor/op_kernel/compressor.cpp` and `csrc/compressor/op_kernel/arch35/**`: verbatim-or-minimally-renamed AscendC A5 kernel and VF dependencies.
- `csrc/CMakeLists.txt`, `include/sgl_kenel_npu_ops.h`, `csrc/pytorch_extensions.cpp`, `csrc/utils/ge_helper.h`: A5-only build, declaration, schema/dispatch, and only helpers proven necessary by tiling.
- `tests/python/sgl_kernel_npu/test_compressor.py`: real-device correctness, state, validation, cache, and graph tests; no CPU result is claimed to validate the AscendC kernel.

### Task 1: Establish the A5-only public contract

**Files:**
- Modify: `include/sgl_kenel_npu_ops.h`
- Modify: `csrc/pytorch_extensions.cpp`
- Test: `tests/python/sgl_kernel_npu/test_compressor.py`

**Consumes:** Existing PrivateUse1 registration pattern in `csrc/pytorch_extensions.cpp`.

**Produces:** `sglang::npu_kernel::compressor(...)` and the public schema:

```cpp
compressor(Tensor x, Tensor wkv, Tensor wgate, Tensor(a!) state_cache,
           Tensor ape, Tensor? state_block_table=None, Tensor? cu_seqlens=None,
           Tensor? seqused=None, Tensor? start_pos=None, *, int cmp_ratio=4,
           int coff=1, int cache_mode=1) -> Tensor
```

- [ ] **Step 1: Write a schema discovery test.**

```python
def test_compressor_schema_is_a5_only():
    if not is_a5():
        self.assertFalse(hasattr(torch.ops.npu, "compressor"))
        return
    schema = torch.ops.npu.compressor.default._schema
    self.assertIn("Tensor(a!) state_cache", str(schema))
    self.assertIn("int cmp_ratio=4", str(schema))
```

- [ ] **Step 2: Run the test before adding registration.**

Run: `python3 tests/python/sgl_kernel_npu/test_compressor.py -k test_compressor_schema_is_a5_only`

Expected: on an A5 wheel without this migration, fail because `torch.ops.npu.compressor` is absent; on non-A5, skip only after confirming the A5 gate is disabled.

- [ ] **Step 3: Add the exact declaration and A5-gated schema/PrivateUse1 implementation.**

```cpp
#ifdef SGL_KERNEL_ENABLE_A5_ONLY_OPS
m.def("compressor(Tensor x, Tensor wkv, Tensor wgate, Tensor(a!) state_cache, "
      "Tensor ape, Tensor? state_block_table=None, Tensor? cu_seqlens=None, "
      "Tensor? seqused=None, Tensor? start_pos=None, *, int cmp_ratio=4, "
      "int coff=1, int cache_mode=1) -> Tensor");
m.impl("compressor", TORCH_FN(sglang::npu_kernel::compressor));
#endif
```

- [ ] **Step 4: Rebuild the A5 wheel, install it, and rerun the schema test.**

Run: `bash build.sh -a kernels && pip install --force-reinstall output/sgl_kernel_npu*.whl && python3 tests/python/sgl_kernel_npu/test_compressor.py -k test_compressor_schema_is_a5_only`

Expected: A5 schema includes the mutable alias; A3/non-A5 build exports no new symbol.

### Task 2: Port the unmodified A5 device and tiling implementation

**Files:**
- Create: `csrc/compressor/op_kernel/compressor.cpp`
- Create: `csrc/compressor/op_kernel/arch35/**`
- Create: `csrc/compressor/op_host/tiling/compressor_tiling.cpp`
- Create: `csrc/compressor/op_host/tiling/compressor_tiling.h`
- Create: `csrc/compressor/op_host/tiling/compressor_tiling_data.h`
- Modify: `csrc/CMakeLists.txt`

**Consumes:** Upstream A5 sources under `D:/GitCode/ops-transformer/attention/compressor`.

**Produces:** An A5 AscendC static target and host tiling code without the upstream OpDef/registration layer.

- [ ] **Step 1: Add a build-source assertion before copying.**

```python
def test_compressor_a5_sources_are_in_kernel_target():
    cmake = Path("csrc/CMakeLists.txt").read_text()
    self.assertIn("compressor/op_kernel/compressor.cpp", cmake)
    self.assertIn("compressor/op_host/tiling/compressor_tiling.cpp", cmake)
```

- [ ] **Step 2: Run it and observe the expected failure.**

Run: `python3 tests/python/sgl_kernel_npu/test_compressor.py -k test_compressor_a5_sources_are_in_kernel_target`

Expected: FAIL because the Compressor sources are not in the project target.

- [ ] **Step 3: Copy only arch35 and shared Compressor files, retaining license headers and source logic.**

```text
Copy op_kernel/compressor.cpp and op_kernel/arch35/**.
Copy op_host/arch35/compressor_tiling.{cpp,h} and all tiling-data headers it includes.
Do not copy arch22 files, OpDef/infer-shape registration, examples, generated OPP files, or tests.
```

- [ ] **Step 4: Add source files and include directories to an A5-only workspace kernel target.**

```cmake
if(SGL_KERNEL_ENABLE_A5_ONLY_OPS)
    list(APPEND OP_SRCS ${PROJECT_OP_SRC_BASE}/compressor/op_host/compressor.cpp
                        ${PROJECT_OP_SRC_BASE}/compressor/op_host/tiling/compressor_tiling.cpp)
    list(APPEND WORKSPACE_KERNEL_SRCS ${PROJECT_OP_SRC_BASE}/compressor/op_kernel/compressor.cpp)
endif()
```

- [ ] **Step 5: Build the A5 kernel target and rerun the source assertion.**

Run: `bash build.sh -a kernels && python3 tests/python/sgl_kernel_npu/test_compressor.py -k test_compressor_a5_sources_are_in_kernel_target`

Expected: build completes on A5+CANN; assertion passes. Record compile errors caused by unavailable CANN APIs rather than replacing synchronization or schedule policy.

### Task 3: Adapt tiling context and implement no-sync input validation

**Files:**
- Create: `csrc/compressor/op_host/compressor.cpp`
- Modify: `csrc/utils/ge_helper.h` only if the imported tiling requires `AttrDef::Bool` or deterministic-level get/set
- Test: `tests/python/sgl_kernel_npu/test_compressor.py`

**Consumes:** Task 1 ABI and Task 2 `CompressorTiling`/`CompressorTilingData`.

**Produces:** Host validation and tiling that derives `stateCacheStrideDim0` from `state_cache.stride(0)` and maps `at::globalContext().deterministicAlgorithms()` to level 3/0.

- [ ] **Step 1: Write parameterized invalid-input tests for host-visible metadata.**

```python
@parameterized.expand([
    ("ring_table_must_be_1d", 2, "state_block_table"),
    ("linear_table_must_be_2d", 1, "state_block_table"),
    ("state_must_be_fp32", 2, "state_cache"),
])
def test_compressor_rejects_invalid_host_metadata(self, name, cache_mode, expected):
    with self.assertRaisesRegex(RuntimeError, expected):
        torch.ops.npu.compressor(*make_invalid_args(name), cmp_ratio=4, coff=1, cache_mode=cache_mode)
```

- [ ] **Step 2: Run the tests and verify failure because the wrapper has no validation.**

Run: `python3 tests/python/sgl_kernel_npu/test_compressor.py -k invalid_host_metadata`

Expected: FAIL by accepting an invalid input or by reaching an unrelated launch error, not because of missing test setup.

- [ ] **Step 3: Implement metadata-only checks and tiling adaptation.**

```cpp
TORCH_CHECK(state_cache.scalar_type() == at::kFloat, "state_cache must be FP32");
TORCH_CHECK(state_cache.dim() == 3 && state_cache.stride(0) > 0,
            "state_cache must be [block_num, block_size, 2*coff*D] with positive dim-0 stride");
const auto state_cache_stride_dim0 = state_cache.stride(0);
const auto deterministic_level = at::globalContext().deterministicAlgorithms() ? 3 : 0;
```

Validate dtype/device/layout/rank/shape/ratio/coff and the Ring/Linear table rank using sizes, dtypes, and devices only. Set `grad_enabled=false`; make ABI outputs private `at::empty` tensors with the native-required shapes and dtypes.

- [ ] **Step 4: Run invalid-input and deterministic tiling tests.**

Run: `python3 tests/python/sgl_kernel_npu/test_compressor.py -k 'invalid_host_metadata or deterministic'`

Expected: all invalid cases raise their named validation error; deterministic-on/off choose distinct cache keys without synchronizing tensor values.

### Task 4: Add graph-safe per-device tiling cache and direct launch

**Files:**
- Modify: `csrc/compressor/op_host/compressor.cpp`
- Test: `tests/python/sgl_kernel_npu/test_compressor.py`

**Consumes:** Validated inputs and tiling data from Task 3.

**Produces:** Mutex-protected per-device cache whose key includes every tiling input and whose device tiling storage survives graph replay.

- [ ] **Step 1: Write cache and graph capture tests.**

```python
def test_compressor_graph_replay_uses_warmed_tiling_and_current_bank_table(self):
    args = make_ring_args(batch=2, banks=[0, 3])
    torch.ops.npu.compressor(*args, cmp_ratio=4, coff=1, cache_mode=2)  # warmup
    captured = capture(lambda: torch.ops.npu.compressor(*args, cmp_ratio=4, coff=1, cache_mode=2))
    args.state_block_table.copy_(torch.tensor([2, 1], device="npu", dtype=torch.int32))
    captured.replay()
    self.assert_only_banks_changed(args.state_cache, [2, 1])
```

- [ ] **Step 2: Run the graph test and confirm it fails before caching exists.**

Run: `python3 tests/python/sgl_kernel_npu/test_compressor.py -k graph_replay_uses_warmed_tiling_and_current_bank_table`

Expected: FAIL due to capture-time allocation/cache miss handling or stale bank behavior.

- [ ] **Step 3: Implement a key and cache entry with device-owned tiling storage.**

```cpp
struct CompressorTilingKey {
    c10::Device device;
    std::array<int64_t, 3> state_cache_sizes;
    std::array<int64_t, 3> x_sizes;
    int64_t state_stride_dim0, cmp_ratio, coff, cache_mode, deterministic_level;
    // Include all required/optional tensor ranks, sizes, dtypes, layouts and template-selecting conditions.
};
```

Guard lookup/build with a mutex. Detect capture using the torch-npu graph-state API already used in this repository; on capture miss `TORCH_CHECK(false, "compressor tiling cache miss during graph capture; warm up first")`. Eager overflow may use an uncached entry, but capture may not.

- [ ] **Step 4: Launch with cached tiling and workspace only after a permitted cache path is selected.**

```cpp
EXEC_KERNEL_CMD(compressor, entry.block_dim, x, wkv, wgate, state_cache, ape,
                state_block_table, cu_seqlens, seqused, start_pos, cmp_kv,
                state_cache_out, softmax_score_out, kv_out, workspace, entry.tiling);
```

- [ ] **Step 5: Run eager, capture-hit, and capture-miss tests.**

Run: `python3 tests/python/sgl_kernel_npu/test_compressor.py -k 'graph or cache'`

Expected: replay uses the updated device `state_block_table`; a cold capture reports the warmup error; neither path allocates tiling storage during capture.

### Task 5: Verify A5 numerical and state semantics

**Files:**
- Create: `tests/python/sgl_kernel_npu/test_compressor.py`

**Consumes:** Built operator from Tasks 1-4 and upstream golden/reference code adapted as a Python test helper.

**Produces:** Hardware tests for BSH/TH, BF16/FP16, Linear/Ring, bank isolation/wrap/stride, and documented prefill-boundary evidence.

- [ ] **Step 1: Add reference-driven Ring state tests.**

```python
def test_ring_bank_zero_and_nonsequential_banks_match_reference(self):
    args = make_ring_args(batch=3, banks=[0, 5, 2], ring_size=8, coff=1, head_dim=512)
    expected_out, expected_state = compressor_reference(*args, cmp_ratio=4, coff=1, cache_mode=2)
    actual_out = torch.ops.npu.compressor(*args, cmp_ratio=4, coff=1, cache_mode=2)
    torch.testing.assert_close(actual_out, expected_out, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(args.state_cache, expected_state, rtol=RTOL, atol=ATOL)
```

- [ ] **Step 2: Run the new test and observe the expected failure before the migration is complete.**

Run: `python3 tests/python/sgl_kernel_npu/test_compressor.py -k ring_bank_zero_and_nonsequential_banks_match_reference`

Expected: FAIL from absent/incomplete implementation or output/state mismatch.

- [ ] **Step 3: Add focused cases without adding production features.**

```text
BSH and TH; BF16 and FP16; D=128 and D=512; coff=1 and coff=2;
cache_mode=1 and cache_mode=2; ring positions ring-1/ring/ring+1/2*ring+k;
non-contiguous valid dim-0 stride; empty token batch; and state initial values
KV=0 / score=-inf.
```

- [ ] **Step 4: Run the complete operator suite on A5 hardware.**

Run: `python3 tests/python/sgl_kernel_npu/test_compressor.py`

Expected: output and final state meet the recorded BF16/FP16 tolerance for every supported case. Preserve the exact CANN version, SoC, command, and result in the test report or handoff.

- [ ] **Step 5: Execute and record the prefill boundary gate.**

Run: `python3 tests/python/sgl_kernel_npu/test_compressor.py -k prefill_boundary`

Expected: compare `S = ring_size - coff*cmp_ratio + 1` and `S + 1` against the reference, including final state and next decode. Until this evidence permits long TH prefill, later SGLang integration must reject unsafe calls.

### Task 6: Regression build gates and review

**Files:**
- Modify: only files from Tasks 1-5 when a test reveals a migration defect

**Consumes:** A5 verification results.

**Produces:** Evidence that the gate does not leak A5 symbols or regress A3 builds.

- [ ] **Step 1: Build in A5 configuration and run Compressor tests.**

Run: `bash build.sh -a kernels && pip install --force-reinstall output/sgl_kernel_npu*.whl && python3 tests/python/sgl_kernel_npu/test_compressor.py`

Expected: successful A5 build and all supported real-device Compressor tests pass.

- [ ] **Step 2: Build a non-A5/A3 configuration and inspect the exported schema.**

Run: `bash build.sh -a kernels && python3 -c "import torch, sgl_kernel_npu; assert not hasattr(torch.ops.npu, 'compressor')"`

Expected: build succeeds without compiling or registering Compressor; existing focused A3 test remains runnable.

- [ ] **Step 3: Run formatting and narrow regression checks.**

Run: `pre-commit run --files csrc/CMakeLists.txt csrc/pytorch_extensions.cpp include/sgl_kenel_npu_ops.h csrc/compressor/op_host/compressor.cpp tests/python/sgl_kernel_npu/test_compressor.py`

Expected: all applicable hooks pass. If the local machine lacks CANN, report that hardware compilation/numerical verification remains unexecuted rather than treating static checks as a substitute.

- [ ] **Step 4: Review the final diff against the constraints.**

Run: `git diff --check; git diff -- csrc/CMakeLists.txt csrc/pytorch_extensions.cpp include/sgl_kenel_npu_ops.h csrc/compressor csrc/utils/ge_helper.h tests/python/sgl_kernel_npu/test_compressor.py`

Expected: every changed production line belongs to A5-only operator migration; no `kv_compress_epilog`, A3 logic, or user-owned untracked file is altered.

## Plan self-review

- Spec coverage: Phase A ABI, A5 source import, no-sync validation, stride derivation, deterministic tiling, graph cache/capture rule, A5 build gate, and hardware/reference checks map to Tasks 1-6.
- Deferred by design: SGLang state pool/backend work is Phase C; prefill production policy follows Task 5 gate; PD remains the independently scoped Phase D.
- Placeholder scan: no implementation placeholders are used; uncertain CANN capture API is explicitly required to follow an existing repository pattern during Task 4 rather than inventing an unverified API.
- Type consistency: the same `compressor` schema and `state_cache` alias are used throughout.
