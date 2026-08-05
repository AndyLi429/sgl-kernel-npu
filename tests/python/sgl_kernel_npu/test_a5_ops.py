"""Tests for the Ascend950 (A5) operator set vendored under csrc/a5_ops/.

Two tiers:

* Schema tests run anywhere. The A5 bindings register unconditionally, so the
  operator signatures are checkable on any machine with the wheel installed.
  These catch the failure mode that actually bites during a port -- a schema
  that drifted from vllm-ascend's, which shows up as a TypeError at the call
  site rather than a build error.
* Numerical tests need an A5 device plus the custom OPP package, and skip
  otherwise.
"""

import unittest

import torch

try:
    import sgl_kernel_npu  # noqa: F401
    import torch_npu  # noqa: F401
    from sgl_kernel_npu import a5_ops

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - environment dependent
    a5_ops = None
    _IMPORT_ERROR = exc


def swiglu_group_quant_ref(x, topk_weight=None, group_size=128):
    """float32 reference for torch.ops.npu.swiglu_group_quant (quant_mode=1).

    Mirrors the kernel's VFSwiGlu: y = silu(x0) * x1 over the first/second half
    of the last dim, optionally scaled by topk_weight, then per-group symmetric
    quantization onto the FP8 E4M3 range.
    """
    x = x.to(torch.float32)
    x0, x1 = x.chunk(2, dim=-1)
    y = (x0 / (1.0 + torch.exp(-x0))) * x1
    if topk_weight is not None:
        y = y * topk_weight.to(torch.float32).reshape(-1, 1)

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    groups = y.reshape(*y.shape[:-1], y.shape[-1] // group_size, group_size)
    scale = groups.abs().amax(dim=-1).clamp(min=1e-12) / fp8_max
    quantized = groups / scale.unsqueeze(-1)
    return quantized.reshape(y.shape), scale


class TestA5OpSchemas(unittest.TestCase):
    """Signature parity with vllm-ascend. Runs without an NPU."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"sgl_kernel_npu not importable: {_IMPORT_ERROR}")

    def test_all_ops_registered(self):
        for name in a5_ops.A5_OPS:
            self.assertTrue(
                hasattr(torch.ops.npu, name),
                f"torch.ops.npu.{name} is not registered; the A5 bindings "
                f"should register regardless of build SOC",
            )

    def test_schemas_match_vllm_ascend(self):
        # Argument names, order and defaults, transcribed from vllm-ascend
        # csrc/torch_binding.cpp at merge 55edd9f. A mismatch here means a
        # caller ported from vllm-ascend would break.
        expected = {
            "indexer_compress_epilog": (
                "indexer_compress_cache",
                "indexer_compress_cache_scale",
                "x",
                "slot_mapping",
                "quant_mode",
                "round_scale",
            ),
            "indexer_compress_epilog_v2": (
                "indexer_compress_cache",
                "x",
                "slot_mapping",
                "layout",
            ),
            "kv_compress_epilog": (
                "kv_compress_cache",
                "x",
                "slot_mapping",
                "quant_group_size",
                "quant_mode",
                "round_scale_flag",
                "layout",
            ),
            "load_index_kv_cache": ("kv_cache", "slot_mapping"),
            "hc_pre_v2": (
                "x",
                "hc_fn",
                "hc_scale",
                "hc_base",
                "hc_mult",
                "hc_sinkhorn_iters",
                "norm_eps",
                "hc_eps",
            ),
            "swiglu_group_quant": (
                "x",
                "topk_weight",
                "group_index",
                "dst_type",
                "quant_mode",
                "group_size",
                "round_scale",
                "ue8m0_scale",
                "output_origin",
                "group_list_type",
                "clamp_value",
            ),
        }
        for name, arg_names in expected.items():
            with self.subTest(op=name):
                schema = getattr(torch.ops.npu, name).default._schema
                actual = tuple(arg.name for arg in schema.arguments)
                self.assertEqual(actual, arg_names)

    def test_sparse_attn_optional_tensor_defaults(self):
        # Every KV/index input on the sparse attention op is optional so a
        # caller can drive either the original-KV or compressed-KV branch.
        schema = torch.ops.npu.kv_quant_sparse_attn_sharedkv.default._schema
        by_name = {arg.name: arg for arg in schema.arguments}
        # q and kv_quant_mode are the only required arguments.
        self.assertFalse(by_name["q"].has_default_value())
        self.assertFalse(by_name["kv_quant_mode"].has_default_value())
        for name in ("ori_kv", "cmp_kv", "ori_block_table", "cmp_block_table"):
            self.assertTrue(
                by_name[name].type.isSubtypeOf(
                    torch.OptionalType(torch.TensorType.get())
                )
            )
        self.assertEqual(by_name["layout_q"].default_value, "BSND")
        self.assertEqual(by_name["layout_kv"].default_value, "PA_ND")
        self.assertEqual(by_name["ori_mask_mode"].default_value, 4)
        self.assertEqual(by_name["cmp_mask_mode"].default_value, 3)
        self.assertEqual(by_name["ori_win_left"].default_value, 127)


class TestA5MetaKernels(unittest.TestCase):
    """Fake-tensor shape propagation. Runs without an NPU.

    Without these the ops cannot be traced under torch.compile / graph mode,
    which is the parity gap vllm-ascend closes in torch_binding_meta.cpp.
    """

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"sgl_kernel_npu not importable: {_IMPORT_ERROR}")

    def test_swiglu_group_quant_meta_shapes(self):
        x = torch.empty(8, 512, dtype=torch.bfloat16, device="meta")
        y, scale, y_origin = torch.ops.npu.swiglu_group_quant(
            x, None, None, quant_mode=1, group_size=128
        )
        self.assertEqual(tuple(y.shape), (8, 256))
        self.assertEqual(y.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(scale.shape), (8, 2))  # 256 / 128
        self.assertEqual(tuple(y_origin.shape), (8, 256))
        self.assertEqual(y_origin.dtype, torch.bfloat16)

    def test_load_index_kv_cache_meta_shapes(self):
        kv_cache = torch.empty(64, 128, dtype=torch.bfloat16, device="meta")
        slot_mapping = torch.empty(4, dtype=torch.int32, device="meta")
        kv, kv_scale = torch.ops.npu.load_index_kv_cache(kv_cache, slot_mapping)
        self.assertEqual(tuple(kv.shape), (4, 128))
        self.assertEqual(kv.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(kv_scale.shape), (4,))

    def test_sparse_attn_meta_shapes(self):
        q = torch.empty(2, 4, 8, 128, dtype=torch.bfloat16, device="meta")
        out, lse = torch.ops.npu.kv_quant_sparse_attn_sharedkv(
            q, 1, return_softmax_lse=True
        )
        self.assertEqual(tuple(out.shape), tuple(q.shape))
        self.assertEqual(tuple(lse.shape), (2, 4, 8, 1))
        self.assertEqual(lse.dtype, torch.float32)

    def test_hc_pre_v2_meta_shapes(self):
        # x is [batch, size, hc, d] with hc == hc_mult
        hc_mult, d = 4, 64
        x = torch.empty(2, 3, hc_mult, d, dtype=torch.bfloat16, device="meta")
        hc_fn = torch.empty(
            (2 + hc_mult) * hc_mult, hc_mult * d, dtype=torch.float32, device="meta"
        )
        hc_scale = torch.empty(3, dtype=torch.float32, device="meta")
        hc_base = torch.empty(
            (2 + hc_mult) * hc_mult, dtype=torch.float32, device="meta"
        )
        y, post, comb_frag = torch.ops.npu.hc_pre_v2(
            x, hc_fn, hc_scale, hc_base, hc_mult, 3, 1e-6, 1e-20
        )
        self.assertEqual(tuple(y.shape), (2, 3, d))
        self.assertEqual(tuple(post.shape), (2, 3, hc_mult))
        self.assertEqual(tuple(comb_frag.shape), (2, 3, hc_mult, hc_mult))


class TestA5OpsOnDevice(unittest.TestCase):
    """Numerical checks. Need an Ascend950 plus the custom OPP package."""

    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"sgl_kernel_npu not importable: {_IMPORT_ERROR}")
        if not a5_ops.is_available():
            self.skipTest(
                "A5 custom operator package not installed; "
                "rebuild with './build.sh -a a5ops'"
            )
        if not torch.npu.is_available():
            self.skipTest("no NPU device available")

    def test_swiglu_group_quant_matches_reference(self):
        torch.manual_seed(0)
        num_tokens, hidden = 32, 512  # last dim divisible by 256 for group quant
        x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16).npu()

        y, scale, _ = torch.ops.npu.swiglu_group_quant(
            x, None, None, quant_mode=1, group_size=128
        )

        ref_y, ref_scale = swiglu_group_quant_ref(x.cpu(), group_size=128)
        torch.testing.assert_close(
            scale.cpu().to(torch.float32),
            ref_scale.reshape(scale.shape),
            rtol=2e-2,
            atol=2e-2,
        )
        torch.testing.assert_close(
            y.cpu().to(torch.float32),
            ref_y,
            rtol=1e-1,
            atol=1.0,  # FP8 E4M3 has ~2 decimal digits of mantissa
        )

    def test_load_index_kv_cache_gathers_rows(self):
        torch.manual_seed(0)
        num_blocks, head_dim = 64, 128
        kv_cache = torch.randn(num_blocks, head_dim, dtype=torch.bfloat16).npu()
        slot_mapping = torch.tensor([3, 0, 17, 63], dtype=torch.int32).npu()

        kv, kv_scale = torch.ops.npu.load_index_kv_cache(kv_cache, slot_mapping)

        self.assertEqual(tuple(kv.shape), (slot_mapping.numel(), head_dim))
        self.assertEqual(tuple(kv_scale.shape), (slot_mapping.numel(),))
        self.assertEqual(kv.dtype, torch.float8_e4m3fn)
        self.assertEqual(kv_scale.dtype, torch.float32)


class TestA5Availability(unittest.TestCase):
    def setUp(self):
        if _IMPORT_ERROR is not None:
            self.skipTest(f"sgl_kernel_npu not importable: {_IMPORT_ERROR}")

    def test_assert_available_message_names_the_build_flag(self):
        if a5_ops.is_available():
            a5_ops.assert_available()  # must not raise
            return
        with self.assertRaises(RuntimeError) as ctx:
            a5_ops.assert_available()
        self.assertIn("-a a5ops", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
