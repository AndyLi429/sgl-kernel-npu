import ctypes
import os
import pathlib
from functools import lru_cache, wraps

import torch
import torch_npu

_A5_VENDOR_SUFFIX = "_transformer"


def _setup_a5_custom_opp():
    """Make the Ascend950 custom operator package visible, if it was built.

    The A5 ops in csrc/a5_ops/ ship as a CANN custom OPP package installed under
    this package's ``vendors/`` directory. Two things have to happen before the
    ops can be called:

    * CANN needs ``ASCEND_CUSTOM_OPP_PATH`` to find the tiling/impl registry.
    * ``libcust_opapi.so`` has to be loadable, because the bindings reach it via
      ``dlopen`` by soname. Extending ``LD_LIBRARY_PATH`` here would be too late
      -- the dynamic loader read it at process start -- so the .so is preloaded
      by absolute path instead, which puts it in the process's loaded set and
      lets the later dlopen resolve against it. libsgl_kernel_npu.so also
      carries an rpath to the same directory; either path alone is sufficient.

    A wheel built without ``-a a5ops`` simply has no vendors/ directory, and
    calling an A5 op then raises a TORCH_CHECK naming the missing entrypoint.
    """
    vendors_dir = pathlib.Path(__file__).parent / "vendors"
    if not vendors_dir.is_dir():
        return

    # The vendor directory name is decided by the run package's VENDOR_NAME at
    # build time, so discover it rather than hard-coding the suffix.
    vendor_dirs = sorted(p for p in vendors_dir.iterdir() if p.is_dir())
    if not vendor_dirs:
        return
    vendor_dir = next(
        (p for p in vendor_dirs if p.name.endswith(_A5_VENDOR_SUFFIX)), vendor_dirs[0]
    )

    os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join(
        p for p in (str(vendor_dir), os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")) if p
    )

    cust_opapi = vendor_dir / "op_api" / "lib" / "libcust_opapi.so"
    if cust_opapi.is_file():
        os.environ["LD_LIBRARY_PATH"] = ":".join(
            p
            for p in (str(cust_opapi.parent), os.environ.get("LD_LIBRARY_PATH", ""))
            if p
        )
        try:
            ctypes.CDLL(str(cust_opapi), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            # Leave it to the rpath: a failure here is only fatal if that also
            # misses, and EXEC_NPU_CMD reports that with the operator name.
            pass


def _load_sgl_kernel_npu():
    npu_path = pathlib.Path(__file__).parents[0]
    so_path = os.path.join(npu_path, "lib", "libsgl_kernel_npu.so")
    torch.ops.load_library(so_path)


_setup_a5_custom_opp()
_load_sgl_kernel_npu()
