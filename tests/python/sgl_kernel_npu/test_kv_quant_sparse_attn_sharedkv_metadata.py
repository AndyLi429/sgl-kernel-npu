"""Smoke tests for the A5 KV-quantized shared-KV metadata op."""

import unittest

import sgl_kernel_npu  # noqa: F401
import pytest
import torch
import torch_npu  # noqa: F401

from utils import require_npu_op


pytestmark = require_npu_op("kv_quant_sparse_attn_sharedkv_metadata")


class TestKvQuantSparseAttnSharedkvMetadata(unittest.TestCase):
    def _call(self, num_heads_q):
        cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device="npu")
        seqused_kv = torch.tensor([128], dtype=torch.int32, device="npu")
        return torch.ops.npu.kv_quant_sparse_attn_sharedkv_metadata(
            num_heads_q,
            1,
            512,
            1,
            cu_seqlens_q,
            None,
            None,
            None,
            seqused_kv,
            1,
            1,
            128,
            0,
            0,
            64,
            64,
            1,
            4,
            3,
            127,
            0,
            "TND",
            "PA_ND",
            True,
            False,
            "npu",
        )

    def test_output_contract(self):
        metadata = self._call(64)
        self.assertEqual(tuple(metadata.shape), (1024,))
        self.assertEqual(metadata.dtype, torch.int32)
        self.assertEqual(metadata.device.type, "npu")
        self.assertGreaterEqual(int(metadata.cpu()[8]), 1)

    def test_n128_duplicates_fa_records(self):
        metadata = self._call(128).cpu().view(36, 9)
        self.assertTrue(torch.equal(metadata[0, :8], metadata[1, :8]))
        self.assertEqual(int(metadata[0, 8]), int(metadata[1, 8]))


if __name__ == "__main__":
    unittest.main()
