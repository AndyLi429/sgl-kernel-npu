import unittest

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

import sgl_kernel_npu  # noqa: F401


HC_MULT = 4
MIX_HC = 24
HC_SINKHORN_ITERS = 20
NORM_EPS = 1e-6
HC_EPS = 1e-6


def to_hf32(tensor):
    return (tensor.contiguous().view(torch.int32) & ~((1 << 13) - 1)).view(torch.float32)


def hc_pre_reference(x, hc_fn, hc_scale, hc_base):
    x_float = x.float()
    x_flat = x_float.flatten(-2)
    inv_rms = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + NORM_EPS)
    mixes = F.linear(to_hf32(x_flat), to_hf32(hc_fn.float())) * inv_rms
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


class TestHcPre(unittest.TestCase):
    def test_hc_pre_matches_reference_for_dsv4_shapes(self):
        """A missing HC projection, post mix, or Sinkhorn normalization fails this test."""
        torch.manual_seed(1024)
        for shape in ((1, HC_MULT, 4096), (1, HC_MULT, 7168), (1, 2, HC_MULT, 4096)):
            with self.subTest(shape=shape):
                hidden_size = shape[-1]
                x = (torch.rand(shape, dtype=torch.float32) * 2).to(torch.bfloat16)
                hc_fn = torch.rand(MIX_HC, HC_MULT * hidden_size, dtype=torch.float32) / (HC_MULT * hidden_size)
                hc_scale = torch.rand(3, dtype=torch.float32) * 2
                hc_base = torch.rand(MIX_HC, dtype=torch.float32) * 2
                expected = hc_pre_reference(x, hc_fn, hc_scale, hc_base)

                actual = torch.ops.npu.npu_hc_pre_v2(
                    x.npu(), hc_fn.npu(), hc_scale.npu(), hc_base.npu(),
                    HC_MULT, HC_SINKHORN_ITERS, NORM_EPS, HC_EPS,
                )

                for result, golden in zip(actual, expected):
                    torch.testing.assert_close(result.cpu().float(), golden.float(), rtol=4e-3, atol=4e-3)


if __name__ == "__main__":
    unittest.main()
