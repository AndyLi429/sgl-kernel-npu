import gc
import unittest

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

import sgl_kernel_npu  # noqa: F401


HC_MULT = 4
DSV4_FLASH_HIDDEN_SIZE = 4096
EXTENDED_HIDDEN_SIZE = 7168
MIX_HC = 24
HC_SINKHORN_ITERS = 20
NORM_EPS = 1e-6
HC_EPS = 1e-6
Y_DIFF_THRESHOLD = 4e-3
Y_REQUIRED_PASS_RATE = 0.98
AUX_DIFF_THRESHOLD = 1e-4
AUX_REQUIRED_PASS_RATE = 0.995


def make_hc_pre_inputs(shape):
    torch.manual_seed(1024)
    hidden_size = shape[-1]
    fan_in = HC_MULT * hidden_size
    x = (torch.rand(shape, dtype=torch.float32) * 2).to(torch.bfloat16)
    hc_fn = torch.rand(MIX_HC, fan_in, dtype=torch.float32) / fan_in
    hc_scale = torch.rand(3, dtype=torch.float32) * 2
    hc_base = torch.rand(MIX_HC, dtype=torch.float32) * 2
    return x, hc_fn, hc_scale, hc_base


def hc_pre_reference(x, hc_fn, hc_scale, hc_base):
    x_float = x.float()
    x_flat = x_float.flatten(-2)
    inv_rms = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + NORM_EPS)
    mixes = F.linear(x_flat, hc_fn) * inv_rms
    pre, post, comb = mixes.split([HC_MULT, HC_MULT, HC_MULT * HC_MULT], dim=-1)
    comb = comb.unflatten(-1, (HC_MULT, HC_MULT))

    pre = torch.sigmoid(pre * hc_scale[0] + hc_base[:HC_MULT]) + HC_EPS
    post = 2 * torch.sigmoid(post * hc_scale[1] + hc_base[HC_MULT : 2 * HC_MULT])
    comb = comb * hc_scale[2] + hc_base[2 * HC_MULT :].view(HC_MULT, HC_MULT)
    comb = comb.softmax(-1) + HC_EPS
    comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(HC_SINKHORN_ITERS - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)

    y = (pre.unsqueeze(-1) * x_float).sum(dim=-2).to(x.dtype)
    return y, post, comb


def assert_close_with_pass_rate(actual, expected, diff_threshold, required_pass_rate):
    actual = actual.cpu().float()
    expected = expected.cpu().float()
    abs_diff = (actual - expected).abs()
    magnitude = torch.maximum(actual.abs(), expected.abs())
    close = (abs_diff <= diff_threshold) | (
        abs_diff / magnitude.clamp_min(torch.finfo(torch.float32).tiny) <= diff_threshold
    )
    pass_rate = close.float().mean().item()
    max_abs_diff = abs_diff.max().item()
    assert pass_rate >= required_pass_rate, (
        f"pass rate {pass_rate:.2%} is below {required_pass_rate:.2%}; "
        f"max absolute difference: {max_abs_diff}"
    )


class TestHcPre(unittest.TestCase):
    def compare_hc_pre_with_reference(self, shape):
        x, hc_fn, hc_scale, hc_base = make_hc_pre_inputs(shape)
        expected_y, expected_post, expected_comb = hc_pre_reference(x, hc_fn, hc_scale, hc_base)
        y, post, comb = torch.ops.npu.npu_hc_pre_v2(
            x.npu(), hc_fn.npu(), hc_scale.npu(), hc_base.npu(),
            HC_MULT, HC_SINKHORN_ITERS, NORM_EPS, HC_EPS,
        )

        batch_shape = shape[:-2]
        self.assertEqual(y.shape, (*batch_shape, shape[-1]))
        self.assertEqual(post.shape, (*batch_shape, HC_MULT))
        self.assertEqual(comb.shape, (*batch_shape, HC_MULT, HC_MULT))
        self.assertEqual(y.dtype, torch.bfloat16)
        self.assertEqual(post.dtype, torch.float32)
        self.assertEqual(comb.dtype, torch.float32)
        assert_close_with_pass_rate(y, expected_y, Y_DIFF_THRESHOLD, Y_REQUIRED_PASS_RATE)
        assert_close_with_pass_rate(post, expected_post, AUX_DIFF_THRESHOLD, AUX_REQUIRED_PASS_RATE)
        assert_close_with_pass_rate(comb, expected_comb, AUX_DIFF_THRESHOLD, AUX_REQUIRED_PASS_RATE)

    def tearDown(self):
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats()

    @torch.inference_mode()
    def test_bf16_3d_input(self):
        self.compare_hc_pre_with_reference((2, HC_MULT, DSV4_FLASH_HIDDEN_SIZE))

    @torch.inference_mode()
    def test_bf16_4d_input(self):
        self.compare_hc_pre_with_reference((1, 2, HC_MULT, DSV4_FLASH_HIDDEN_SIZE))

    @torch.inference_mode()
    def test_bf16_dsv4_flash_hidden_size(self):
        self.compare_hc_pre_with_reference((4, HC_MULT, DSV4_FLASH_HIDDEN_SIZE))

    @torch.inference_mode()
    def test_bf16_extended_hidden_size(self):
        self.compare_hc_pre_with_reference((2, HC_MULT, EXTENDED_HIDDEN_SIZE))


if __name__ == "__main__":
    unittest.main()
