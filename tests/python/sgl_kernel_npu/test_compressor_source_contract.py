import re
import unittest
from pathlib import Path


class TestCompressorSourceContract(unittest.TestCase):
    def test_direct_launch_uses_batch_schedule_mode(self):
        source = Path("csrc/compressor/op_kernel/compressor.cpp").read_text()
        self.assertRegex(
            source,
            re.compile(r"__schedmode__\s*\(\s*1\s*\)\s*__global__"),
        )

    def test_tiling_copy_is_checked_before_kernel_launch(self):
        source = Path("csrc/compressor/op_host/compressor.cpp").read_text()
        result_position = source.index("const auto copyStatus =")
        copy_position = source.index("aclrtMemcpy", result_position)
        check_position = source.index("copyStatus == ACL_SUCCESS", copy_position)
        launch_position = source.index("EXEC_KERNEL_CMD(compressor", check_position)
        self.assertLess(result_position, copy_position)
        self.assertLess(copy_position, check_position)
        self.assertLess(check_position, launch_position)


if __name__ == "__main__":
    unittest.main()
