import ast
from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python"
    / "sgl_kernel_npu"
    / "sgl_kernel_npu"
    / "mamba"
    / "causal_conv1d.py"
)


def _module_tree() -> ast.Module:
    return ast.parse(SOURCE.read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def test_prepare_data_accepts_host_shape_metadata():
    prepare_data = _function(_module_tree(), "prepare_data")
    args = {arg.arg for arg in prepare_data.args.args}

    assert {
        "query_start_loc_cpu",
        "max_seqlen",
        "cu_seq_len",
    }.issubset(args)


def test_causal_conv1d_fn_forwards_host_shape_metadata_to_prepare_data():
    tree = _module_tree()
    wrapper = _function(tree, "causal_conv1d_fn_npu")
    wrapper_args = {arg.arg for arg in wrapper.args.args}

    assert {
        "query_start_loc_cpu",
        "max_seqlen",
        "cu_seq_len",
    }.issubset(wrapper_args)

    prepare_call = None
    for node in ast.walk(wrapper):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "prepare_data"
        ):
            prepare_call = node
            break
    assert prepare_call is not None

    forwarded_keywords = {keyword.arg for keyword in prepare_call.keywords}
    assert {
        "query_start_loc_cpu",
        "max_seqlen",
        "cu_seq_len",
    }.issubset(forwarded_keywords)


def test_causal_conv1d_fn_accepts_max_t_alias_from_kwargs():
    wrapper = _function(_module_tree(), "causal_conv1d_fn_npu")

    for node in ast.walk(wrapper):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "kwargs"
            and node.func.attr == "pop"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "max_T"
        ):
            return

    raise AssertionError("causal_conv1d_fn_npu must accept max_T as a kwargs alias")
