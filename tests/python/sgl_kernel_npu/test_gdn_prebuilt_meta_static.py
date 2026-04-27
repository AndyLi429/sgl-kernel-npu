import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FLA_DIR = ROOT / "python" / "sgl_kernel_npu" / "sgl_kernel_npu" / "fla"


def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _arg_names(fn: ast.FunctionDef) -> set[str]:
    return {
        arg.arg
        for arg in [
            *fn.args.posonlyargs,
            *fn.args.args,
            *fn.args.kwonlyargs,
        ]
    }


def _source_contains(path: Path, needle: str) -> bool:
    return needle in path.read_text(encoding="utf-8")


class TestGDNPrebuiltMetaStatic(unittest.TestCase):
    def test_top_level_gdn_kernel_accepts_and_forwards_prebuilt_meta(self):
        tree = _module(FLA_DIR / "chunk.py")

        public_fn = _function(tree, "chunk_gated_delta_rule_npu")
        fwd_fn = _function(tree, "chunk_gated_delta_rule_fwd")

        self.assertIn("prebuilt_meta", _arg_names(public_fn))
        self.assertIn("prebuilt_meta", _arg_names(fwd_fn))
        self.assertTrue(
            _source_contains(FLA_DIR / "chunk.py", "prebuilt_meta=prebuilt_meta"),
            "chunk_gated_delta_rule_npu should pass prebuilt_meta into fwd path",
        )

    def test_gdn_substeps_accept_prebuilt_chunk_metadata(self):
        expected_args = {
            "cumsum.py": ("block_indices",),
            "chunk_scaled_dot_kkt.py": ("chunk_indices",),
            "solve_tril.py": ("chunk_indices_large_block", "chunk_indices_bt"),
            "wy_fast.py": ("chunk_indices",),
            "chunk_delta_h.py": ("chunk_indices", "chunk_offsets"),
            "chunk_o.py": ("chunk_indices", "chunk_offsets"),
        }
        function_names = {
            "cumsum.py": "chunk_local_cumsum_scalar_npu",
            "chunk_scaled_dot_kkt.py": "chunk_scaled_dot_kkt_fwd_npu",
            "solve_tril.py": "solve_tril_npu",
            "wy_fast.py": "recompute_w_u_fwd_npu",
            "chunk_delta_h.py": "chunk_gated_delta_rule_fwd_h_npu",
            "chunk_o.py": "chunk_fwd_o_npu",
        }

        for filename, args in expected_args.items():
            with self.subTest(filename=filename):
                fn = _function(_module(FLA_DIR / filename), function_names[filename])
                names = _arg_names(fn)
                for arg in args:
                    self.assertIn(arg, names)


if __name__ == "__main__":
    unittest.main()
