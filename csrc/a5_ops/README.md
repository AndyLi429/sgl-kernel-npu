# Ascend950 (A5) custom operators

Vendored from [vllm-ascend](https://github.com/vllm-project/vllm-ascend) PR
[#9271](https://github.com/vllm-project/vllm-ascend/pull/9271) at merge commit
`55edd9f`, which adds the A5 / `ascend950` operator set behind the DeepSeek V4
A5 path.

## Why this is a separate tree

The ops in `csrc/<op>/` are *direct-launch*: the host side computes tiling
inline and fires `EXEC_KERNEL_CMD`, and the device code is compiled straight
into `libsgl_kernel_npu.so`. The A5 ops instead use the CANN **custom operator
(OPP)** model — `OpDef`/`OP_ADD` registration, `_proto.cpp` shape inference,
`IMPL_OP_OPTILING` tiling — and are reached through generated `aclnn*` APIs.

Rather than rewrite them into the direct-launch style, this directory keeps the
upstream `cann-ops-transformer` build framework (`build.sh`, `cmake/`,
`common/`). That is deliberate: each op's `op_host/CMakeLists.txt` carries
compile options the generated kernel binaries depend on —

```cmake
add_ops_compile_options(
        OP_NAME IndexerCompressEpilog
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
)
```

— so re-expressing the build would risk silent performance divergence from
vllm-ascend. The vendored ops have no cross-directory `#include`s, which makes
this tree self-contained.

The same reasoning is why `csrc/deepep/` already builds its own OPP package;
this is the second such package in the repo, not a new pattern.

## Layout

```
csrc/a5_ops/
├── build.sh            # vendored cann-ops-transformer build framework
├── CMakeLists.txt      # vendored op-project top level
├── cmake/              # vendored (72 files)
├── common/             # vendored shared headers the framework expects
├── attention/          # 6 ops
├── moe/                # 2 ops
└── build_a5_ops.sh     # thin wrapper: build + install into the wheel
```

## Operators

| Directory | `torch.ops.npu.*` | aclnn entrypoint |
| --- | --- | --- |
| `attention/indexer_compress_epilog` | `indexer_compress_epilog` | `aclnnIndexerCompressEpilog` |
| `attention/indexer_compress_epilog_v2` | `indexer_compress_epilog_v2` | `aclnnIndexerCompressEpilogV2` |
| `attention/kv_compress_epilog` | `kv_compress_epilog` | `aclnnKvCompressEpilog` |
| `attention/load_index_kv_cache` | `load_index_kv_cache` | `aclnnLoadIndexKvCache` |
| `attention/kv_quant_sparse_attn_sharedkv` | `kv_quant_sparse_attn_sharedkv` | `aclnnKvQuantSparseAttnSharedkv` |
| `attention/kv_quant_sparse_attn_sharedkv_metadata` | `kv_quant_sparse_attn_sharedkv_metadata` | `aclnnKvQuantSparseAttnSharedkvMetadata` |
| `moe/hc_pre` | `hc_pre_v2` | `aclnnHcPre` |
| `moe/swiglu_group_quant` | `swiglu_group_quant` | `aclnnSwigluGroupQuant` |

`kv_quant_sparse_attn_sharedkv_metadata` is an **AICPU** operator
(`op_kernel_aicpu/`); the vendored framework handles it via `ENABLE_AICPU`.

Schemas match vllm-ascend argument for argument, including defaults. Only the
redundant `npu_` prefix is dropped, since these live under `torch.ops.npu`
already. Bindings are in `csrc/a5_ops_binding/a5_ops.cpp`, which registers both
a `PrivateUse1` implementation and a `Meta` one (ported from vllm-ascend's
`csrc/torch_binding_meta.cpp`) so the ops trace under torch.compile / graph mode
rather than falling back to eager.

## Build

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
./build.sh -a a5ops          # or: ./build.sh Ascend950  (builds everything incl. A5)
pip install output/sgl_kernel_npu*.whl
```

`build_a5_ops.sh` produces `cann-ops-transformer*.run` and installs it into
`python/sgl_kernel_npu/sgl_kernel_npu/vendors/`. `sgl_kernel_npu/__init__.py`
points `ASCEND_CUSTOM_OPP_PATH` at that directory and preloads
`libcust_opapi.so` before loading `libsgl_kernel_npu.so`.

Feature-detect from Python with `sgl_kernel_npu.a5_ops.is_available()`.

Requires an A5 CANN toolkit. Note `cmake/scripts/util/const_var.py` maps
`ascend950` to the platform name **`Ascend950PR_9599`**, not `Ascend950`.

## Not vendored

PR #9271's `ascend950` build list also names ops that already existed on
vllm-ascend main and are **not** part of this port:

`compressor`, `quant_lightning_indexer`, `quant_lightning_indexer_metadata`,
`moe_gating_top_k_hash`, `inplace_partial_rotary_mul`, `hc_pre_sinkhorn`,
`hc_pre_inv_rms`, `hc_post`.

Two consequences worth knowing before wiring up an end-to-end A5 sparse
attention path:

* **`quant_lightning_indexer` is missing.** PR #9271 only *patches* it (5 lines
  making the arch35 PA-cache path stride-aware, plus `AutoContiguous()` →
  `IgnoreContiguous()` on `key`/`key_scale`). This repo currently has the
  unquantized `lightning_indexer` in `csrc/lightning_indexer/`. Porting the
  quantized A5 indexer is separate work.
* **`grouped_matmul_swiglu_quant_weight_nz`** gets a torch binding in PR #9271
  but its kernel lives under vllm-ascend's `csrc/gmm/`, also outside this port.
