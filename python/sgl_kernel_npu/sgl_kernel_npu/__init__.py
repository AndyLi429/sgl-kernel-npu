import os
import pathlib
from functools import lru_cache, wraps

import torch
import torch_npu

_opp_path = str(pathlib.Path(__file__).parents[0] / "ops" / "vendors" / "aie_ascendc")
_custom_opp_paths = [path for path in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":") if path]
if _opp_path not in _custom_opp_paths:
    _custom_opp_paths.append(_opp_path)
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = ":".join(_custom_opp_paths)


def _load_sgl_kernel_npu():
    npu_path = pathlib.Path(__file__).parents[0]
    so_path = os.path.join(npu_path, "lib", "libsgl_kernel_npu.so")
    torch.ops.load_library(so_path)


_load_sgl_kernel_npu()
