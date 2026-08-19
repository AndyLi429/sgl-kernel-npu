# Task 2 Report: A5 Compressor device and tiling port

## Implementation

- Ported the shared Compressor device entrypoint and all `arch35` device
  headers, including the VF helpers, from
  `D:/GitCode/ops-transformer` at `5520571a41e5ff13adf3310a7f25f054881ea099`.
  The entrypoint retains only its upstream A5 branch, so the port has no
  `arch22` source-selection or implementation dependency.  The A5 template
  selection, grad-enabled argument, math, and synchronization remain intact.
- Ported the A5 host tiling implementation and headers.  The local tiling-data
  header is the upstream A5 data layout; the host header uses that local copy.
  No OpDef, infer-shape, registration, OPP, sample, or upstream-test source was
  imported.
- Added the actual Task 2 sources to the host and workspace-kernel source lists
  only under the existing `SGL_KERNEL_ENABLE_A5_ONLY_OPS` guard.  The brief also
  named `compressor/op_host/compressor.cpp`, but that file belongs to the
  deferred Task 3 wrapper and does not exist yet; it is deliberately not listed
  here so this staged change remains buildable.
- Added the required source-list assertion to the existing Compressor test.

## RED/GREEN evidence

The new source-list assertion was added before the port.  The prescribed test
command could not collect on this workstation because `torch_npu` is absent;
the dependency-free equivalent assertion against `csrc/CMakeLists.txt` then
failed as expected before the sources were added.  After the port, that
dependency-free assertion passes.

## Verification

Passed:

```text
python -c "...assert compressor sources are listed in csrc/CMakeLists.txt..."
python -m py_compile tests/python/sgl_kernel_npu/test_compressor.py
git diff --check
pre-commit run --files <Task 2 CMake, Compressor sources, and test>
```

The hooks passed.  Their standard whitespace/clang-format cleanup did not
change any A5 algorithm, template selection, or synchronization behavior.

Blocked:

```text
PYTHONPATH=python/sgl_kernel_npu python tests/python/sgl_kernel_npu/test_compressor.py -k test_compressor_a5_sources_are_in_kernel_target
```

The test cannot collect because this workstation has no `torch_npu` module.

```text
bash build.sh -a kernels
```

The build stops before configure with `Cannot find an Ascend toolkit directory
containing set_env.sh` (SOC defaults to `Ascend910_9382`).  No A5+CANN build,
installed wheel, or hardware run was claimed.

## Concerns

- Task 3 must add the deferred Torch host wrapper and then append its source to
  `OP_SRCS` under the same A5 gate.
- A real Ascend950 CANN environment must rebuild the kernels and run the
  focused Compressor test before this port is considered hardware-validated.
