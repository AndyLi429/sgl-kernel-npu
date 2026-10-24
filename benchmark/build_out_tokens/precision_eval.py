"""Run the exact-integer NPU regression suite and retain its test counts."""

import sys
import unittest
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "tests/python/sgl_kernel_npu")
)

from test_build_out_tokens import TestBuildOutTokens

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestBuildOutTokens)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
