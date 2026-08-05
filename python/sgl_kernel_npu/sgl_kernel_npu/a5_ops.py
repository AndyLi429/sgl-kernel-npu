"""Feature detection for the Ascend950 (A5) operator set.

The A5 ops are registered as ``torch.ops.npu.*`` unconditionally -- their aclnn
entrypoints are resolved lazily at call time -- so the schemas exist even on a
wheel built for A2/A3 or without ``./build.sh -a a5ops``. Calling one there
raises a ``RuntimeError`` from EXEC_NPU_CMD naming the missing entrypoint.

Use :func:`is_available` to branch before dispatching to an A5 path, and
:func:`assert_available` to fail early with a message that says what to build.

Operator names match vllm-ascend's schemas argument for argument; only the
redundant ``npu_`` prefix is dropped, since these already live under
``torch.ops.npu``:

===================================  ==========================================
sgl-kernel-npu                       vllm-ascend
===================================  ==========================================
indexer_compress_epilog              indexer_compress_epilog
indexer_compress_epilog_v2           indexer_compress_epilog_v2
kv_compress_epilog                   kv_compress_epilog
load_index_kv_cache                  npu_load_index_kv_cache
kv_quant_sparse_attn_sharedkv        npu_kv_quant_sparse_attn_sharedkv
kv_quant_sparse_attn_sharedkv_metadata  npu_kv_quant_sparse_attn_sharedkv_metadata
hc_pre_v2                            npu_hc_pre_v2
swiglu_group_quant                   npu_swiglu_group_quant
===================================  ==========================================
"""

import os
import pathlib

A5_OPS = (
    "indexer_compress_epilog",
    "indexer_compress_epilog_v2",
    "kv_compress_epilog",
    "load_index_kv_cache",
    "kv_quant_sparse_attn_sharedkv",
    "kv_quant_sparse_attn_sharedkv_metadata",
    "hc_pre_v2",
    "swiglu_group_quant",
)


def custom_opp_path():
    """Return the installed A5 custom OPP vendor directory, or None."""
    vendors_dir = pathlib.Path(__file__).parent / "vendors"
    if not vendors_dir.is_dir():
        return None
    for vendor_dir in sorted(vendors_dir.iterdir()):
        if (vendor_dir / "op_api" / "lib" / "libcust_opapi.so").is_file():
            return vendor_dir
    return None


def is_available():
    """True when the A5 custom operator package is installed in this wheel.

    This checks that the package was built and shipped, not that the current
    device is an Ascend950 -- an A5 wheel used on A2/A3 still reports True and
    fails at call time inside the kernel.
    """
    return custom_opp_path() is not None


def assert_available():
    if is_available():
        return
    raise RuntimeError(
        "The Ascend950 (A5) custom operator package is not installed in this "
        "sgl_kernel_npu wheel. Rebuild with './build.sh -a a5ops' on an A5 CANN "
        "toolkit (or './build.sh Ascend950'), then reinstall the wheel. "
        f"Looked under {pathlib.Path(__file__).parent / 'vendors'}."
    )
