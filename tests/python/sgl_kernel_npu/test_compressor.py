from pathlib import Path

import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu  # noqa: F401
from torch_npu.testing.testcase import TestCase, run_tests


def is_a5() -> bool:
    return torch.npu.is_available() and torch.npu.get_device_name().lower().startswith(
        "ascend950"
    )


def make_invalid_args(name):
    device = torch.device("npu")
    x = torch.empty((2, 8, 1024), dtype=torch.float16, device=device)
    wkv = torch.empty((128, 1024), dtype=torch.float16, device=device)
    wgate = torch.empty((128, 1024), dtype=torch.float16, device=device)
    state_dtype = torch.float16 if name == "state_must_be_fp32" else torch.float32
    state_cache = torch.empty((4, 128, 256), dtype=state_dtype, device=device)
    ape = torch.empty((4, 128), dtype=torch.float32, device=device)
    if name == "linear_table_must_be_2d":
        state_block_table = torch.empty((2,), dtype=torch.int32, device=device)
    else:
        state_block_table = torch.empty((2, 1), dtype=torch.int32, device=device)
    return x, wkv, wgate, state_cache, ape, state_block_table


class TestCompressorSchema(TestCase):
    def test_compressor_a5_sources_are_in_kernel_target(self):
        cmake = Path("csrc/CMakeLists.txt").read_text()
        self.assertIn("compressor/op_kernel/compressor.cpp", cmake)
        self.assertIn("compressor/op_host/compressor.cpp", cmake)
        self.assertIn("compressor/op_host/tiling/compressor_tiling.cpp", cmake)

    def test_compressor_deterministic_mode_is_forwarded_to_tiling(self):
        source = Path("csrc/compressor/op_host/compressor.cpp").read_text()
        self.assertIn("at::globalContext().deterministicAlgorithms() ? 3 : 0", source)
        self.assertIn("SetDeterministicLevel(deterministic_level)", source)
        self.assertNotIn(".item()", source)
        self.assertNotIn("_local_scalar_dense", source)

    def test_compressor_schema_is_a5_only(self):
        if not is_a5():
            self.assertFalse(hasattr(torch.ops.npu, "compressor"))
            return

        schema = str(torch.ops.npu.compressor.default._schema)
        self.assertIn("Tensor(a!) state_cache", schema)
        self.assertIn("int cmp_ratio=4", schema)
        self.assertIn("int coff=1", schema)
        self.assertIn("int cache_mode=1", schema)
        self.assertNotIn("stride", schema)

    def test_compressor_rejects_invalid_host_metadata(self):
        if not is_a5():
            self.skipTest("compressor is A5-only")

        cases = [
            ("ring_table_must_be_1d", 2, "state_block_table"),
            ("linear_table_must_be_2d", 1, "state_block_table"),
            ("state_must_be_fp32", 2, "state_cache"),
        ]
        for name, cache_mode, expected in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError, expected):
                    torch.ops.npu.compressor(
                        *make_invalid_args(name),
                        cmp_ratio=4,
                        coff=1,
                        cache_mode=cache_mode,
                    )


if __name__ == "__main__":
    run_tests()
