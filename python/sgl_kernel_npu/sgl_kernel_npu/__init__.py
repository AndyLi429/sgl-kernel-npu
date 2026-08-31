import os
import pathlib
from functools import lru_cache, wraps

import torch
import torch_npu

_opp_path = pathlib.Path(__file__).parents[0] / "ops" / "vendors" / "aie_ascendc"
os.environ["ASCEND_CUSTOM_OPP_PATH"] = f"{_opp_path}:{os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')}"
_op_api_path = _opp_path / "op_api" / "lib"
os.environ["LD_LIBRARY_PATH"] = f"{_op_api_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"


def _load_sgl_kernel_npu():
    npu_path = pathlib.Path(__file__).parents[0]
    so_path = os.path.join(npu_path, "lib", "libsgl_kernel_npu.so")
    torch.ops.load_library(so_path)


_load_sgl_kernel_npu()
