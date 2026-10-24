import itertools
import unittest

import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.sample import build_out_tokens


def torch_reference(
    *, draft_tokens, correct_len, bonus, verify_num_draft_tokens, gamma
):
    out = torch.empty(
        (draft_tokens.shape[0], verify_num_draft_tokens),
        dtype=torch.int64,
        device=draft_tokens.device,
    )
    out[:, :gamma].copy_(draft_tokens)
    out[:, gamma].fill_(0)
    out.scatter_(1, correct_len.to(torch.int64)[:, None], bonus[:, None])
    return out


def make_inputs(
    bs, gamma, token_dtype=torch.int64, len_dtype=torch.int32, strided=False
):
    # Generate on CPU for reproducibility across torch_npu versions.
    generator = torch.Generator().manual_seed(bs * 31 + gamma)
    step = 2 if strided else 1
    draft = torch.randint(
        -131072, 131072, (bs, gamma * step), dtype=token_dtype, generator=generator
    )
    bonus = torch.randint(
        -131072, 131072, (bs * step,), dtype=token_dtype, generator=generator
    )
    if token_dtype == torch.int64:
        draft += 2**40
        bonus -= 2**40
    lengths = (torch.arange(bs * step) % (gamma + 1)).to(len_dtype)
    return {
        "draft_tokens": draft.npu()[:, ::step],
        "correct_len": lengths.npu()[::step],
        "bonus": bonus.npu()[::step],
        "verify_num_draft_tokens": gamma + 1,
        "gamma": gamma,
    }


class TestBuildOutTokens(unittest.TestCase):
    def test_exact(self):
        cases = itertools.product(
            (0, 1, 2, 7, 31, 32, 33, 127, 512, 2049),
            (0, 1, 3, 7, 8, 15, 31),
            (torch.int32, torch.int64),
            (torch.int32, torch.int64),
            (False, True),
        )
        for bs, gamma, token_dtype, len_dtype, strided in cases:
            with self.subTest(
                bs=bs,
                gamma=gamma,
                tokens=token_dtype,
                lengths=len_dtype,
                strided=strided,
            ):
                inputs = make_inputs(bs, gamma, token_dtype, len_dtype, strided)
                # Torch scatter requires src and destination to share dtype.
                reference_inputs = dict(inputs, bonus=inputs["bonus"].to(torch.int64))
                expected = torch_reference(**reference_inputs)
                actual = build_out_tokens(**inputs)
                self.assertEqual(actual.dtype, torch.int64)
                self.assertTrue(torch.equal(actual.cpu(), expected.cpu()))

    def test_acceptance_boundaries(self):
        for gamma in (1, 7, 31):
            for length in (0, gamma):
                inputs = make_inputs(33, gamma)
                inputs["correct_len"].fill_(length)
                self.assertTrue(
                    torch.equal(
                        build_out_tokens(**inputs).cpu(),
                        torch_reference(**inputs).cpu(),
                    )
                )

    def test_invalid_width(self):
        inputs = make_inputs(2, 7)
        inputs["verify_num_draft_tokens"] = 9
        with self.assertRaises(ValueError):
            build_out_tokens(**inputs)


if __name__ == "__main__":
    unittest.main()
