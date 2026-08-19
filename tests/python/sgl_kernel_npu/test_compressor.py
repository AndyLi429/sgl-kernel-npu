import sgl_kernel_npu  # noqa: F401
import torch
import torch_npu  # noqa: F401
from torch_npu.testing.testcase import TestCase, run_tests


def is_a5() -> bool:
    return torch.npu.is_available() and torch.npu.get_device_name().lower().startswith(
        "ascend950"
    )


class TestCompressorSchema(TestCase):
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


if __name__ == "__main__":
    run_tests()
