// Copyright (c) 2026 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// torch.ops.npu bindings for the Ascend950 (A5) operator set vendored under
// csrc/a5_ops/. Unlike the direct-launch ops in csrc/<op>/, these reach the
// device through the custom OPP package's generated aclnn entrypoints.
//
// EXEC_NPU_CMD resolves every aclnn symbol through dlopen/dlsym on
// libcust_opapi.so, so this file compiles and registers on any SOC. On a
// non-A5 build the schemas still exist and a call raises a TORCH_CHECK naming
// the missing entrypoint rather than failing to load the library.
//
// Schemas mirror vllm-ascend's (PR #9271) argument order, types and defaults
// exactly so callers port over unchanged; only the redundant `npu_` prefix is
// dropped, since these already live in the `npu` namespace.

#include <torch/extension.h>
#include <torch/library.h>

#include "acl/acl.h"
#include "aclnn_torch_adapter/op_api_common.h"

namespace sglang {
namespace a5_ops {

// ---------------------------------------------------------------- indexer/kv
// cache write + read epilogues

void indexer_compress_epilog(at::Tensor &indexer_compress_cache, at::Tensor &indexer_compress_cache_scale,
                             const at::Tensor &x, const at::Tensor &slot_mapping, int64_t quant_mode, bool round_scale)
{
    EXEC_NPU_CMD(aclnnIndexerCompressEpilog, indexer_compress_cache, indexer_compress_cache_scale, x, slot_mapping,
                 quant_mode, round_scale);
}

void indexer_compress_epilog_v2(at::Tensor &indexer_compress_cache, const at::Tensor &x, const at::Tensor &slot_mapping,
                                int64_t layout)
{
    int64_t indexer_compress_cache_stride = indexer_compress_cache.stride(0);
    EXEC_NPU_CMD(aclnnIndexerCompressEpilogV2, indexer_compress_cache, x, slot_mapping, layout,
                 indexer_compress_cache_stride);
}

void validate_kv_compress_epilog_inputs(const at::Tensor &x, const at::Tensor &slot_mapping,
                                        at::Tensor &kv_compress_cache)
{
    TORCH_CHECK(x.dim() == 2, "x must be 2D tensor, but got dimensions: ", x.dim());
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0, "x dimensions must be positive, but got: [", x.size(0), ", ", x.size(1),
                "]");
    TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be 1D tensor, but got dimensions: ", slot_mapping.dim());
    TORCH_CHECK(slot_mapping.size(0) == x.size(0),
                "slot_mapping size must equal x's first dimension, but got slot_mapping_size=", slot_mapping.size(0),
                ", x.dim(0)=", x.size(0));
    if (kv_compress_cache.dim() == 4) {
        TORCH_CHECK(kv_compress_cache.size(2) == 1,
                    "kv_compress_cache 4D tensor requires headnum (dim 2) == 1, but got ", kv_compress_cache.size(2));
    }
    TORCH_CHECK(x.dtype() == at::kBFloat16, "x must be BF16, but got ", x.dtype());
    TORCH_CHECK(slot_mapping.dtype() == at::kInt || slot_mapping.dtype() == at::kLong,
                "slot_mapping must be INT32 or INT64, but got ", slot_mapping.dtype());
    TORCH_CHECK(kv_compress_cache.dtype() == at::ScalarType::Float8_e5m2 ||
                    kv_compress_cache.dtype() == at::ScalarType::Float8_e4m3fn,
                "kv_compress_cache must be FP8_E5M2 or FP8_E4M3, but got ", kv_compress_cache.dtype());
}

void kv_compress_epilog(at::Tensor &kv_compress_cache, const at::Tensor &x, const at::Tensor &slot_mapping,
                        int64_t quant_group_size, int64_t quant_mode, bool round_scale_flag, int64_t layout)
{
    validate_kv_compress_epilog_inputs(x, slot_mapping, kv_compress_cache);

    at::Tensor cache = kv_compress_cache;
    if (cache.dim() == 4) {
        cache = cache.squeeze(2);
    }

    int64_t round_scale = round_scale_flag ? 1 : 0;
    int64_t cache_stride = cache.stride(0);
    EXEC_NPU_CMD(aclnnKvCompressEpilog, cache, x, slot_mapping, quant_group_size, quant_mode, round_scale, layout,
                 cache_stride);
}

std::tuple<at::Tensor, at::Tensor> construct_load_index_kv_cache_output_tensor(const at::Tensor &kv_cache,
                                                                               const at::Tensor &slot_mapping)
{
    constexpr int64_t KV_LAST_DIM = 128;
    int64_t n = slot_mapping.size(0);

    at::Tensor kv = at::empty({n, KV_LAST_DIM}, kv_cache.options().dtype(at::kFloat8_e4m3fn));
    at::Tensor kv_scale = at::empty({n}, kv_cache.options().dtype(at::kFloat));

    return std::tuple<at::Tensor, at::Tensor>(kv, kv_scale);
}

std::tuple<at::Tensor, at::Tensor> load_index_kv_cache(const at::Tensor &kv_cache, const at::Tensor &slot_mapping)
{
    auto output_tensors = construct_load_index_kv_cache_output_tensor(kv_cache, slot_mapping);
    at::Tensor kv = std::get<0>(output_tensors);
    at::Tensor kv_scale = std::get<1>(output_tensors);

    int64_t kv_cache_stride = kv_cache.stride(0);
    EXEC_NPU_CMD(aclnnLoadIndexKvCache, kv_cache, slot_mapping, kv_cache_stride, kv, kv_scale);

    return std::tuple<at::Tensor, at::Tensor>(kv, kv_scale);
}

// ------------------------------------------------- quantized sparse attention

std::tuple<at::Tensor, at::Tensor> construct_sparse_attn_output_tensor(const at::Tensor &q, bool return_softmax_lse)
{
    for (size_t i = 0; i < q.sizes().size(); i++) {
        TORCH_CHECK(q.size(i) > 0, "All values within query's shape should be greater than 0, but shape[", i, "] is ",
                    q.size(i));
    }
    at::Tensor output = at::empty(q.sizes(), q.options().dtype(q.dtype()));
    at::Tensor softmax_lse;
    if (return_softmax_lse) {
        std::vector<int64_t> lse_sizes(q.sizes().begin(), q.sizes().end());
        lse_sizes.back() = 1;
        softmax_lse = at::empty(lse_sizes, q.options().dtype(c10::ScalarType::Float));
    } else {
        softmax_lse = at::empty({0}, q.options().dtype(c10::ScalarType::Float));
    }
    return std::tuple<at::Tensor, at::Tensor>(output, softmax_lse);
}

std::tuple<at::Tensor, at::Tensor> kv_quant_sparse_attn_sharedkv(
    const at::Tensor &q, int64_t kv_quant_mode, const c10::optional<at::Tensor> &ori_kv,
    const c10::optional<at::Tensor> &cmp_kv, const c10::optional<at::Tensor> &ori_sparse_indices,
    const c10::optional<at::Tensor> &cmp_sparse_indices, const c10::optional<at::Tensor> &ori_block_table,
    const c10::optional<at::Tensor> &cmp_block_table, const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_ori_kv, const c10::optional<at::Tensor> &cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &sinks, const c10::optional<at::Tensor> &metadata, int64_t tile_size,
    int64_t rope_head_dim, double softmax_scale, int64_t cmp_ratio, int64_t ori_mask_mode, int64_t cmp_mask_mode,
    int64_t ori_win_left, int64_t ori_win_right, c10::string_view layout_q, c10::string_view layout_kv,
    bool return_softmax_lse)
{
    std::string layout_q_str = std::string(layout_q);
    std::string layout_kv_str = std::string(layout_kv);
    auto output = construct_sparse_attn_output_tensor(q, return_softmax_lse);
    at::Tensor attn_out = std::get<0>(output);
    at::Tensor softmax_lse = std::get<1>(output);

    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());
    // The A5 kernel indexes the paged KV caches by dim0 stride rather than
    // recomputing block_size * head_num * head_dim, so a non-contiguous cache
    // view stays valid. Passing 0 for an absent cache matches the host check.
    int64_t ori_kv_stride0 = 0;
    int64_t cmp_kv_stride0 = 0;
    if (ori_kv.has_value() && ori_kv.value().defined()) {
        ori_kv_stride0 = ori_kv.value().stride(0);
    }
    if (cmp_kv.has_value() && cmp_kv.value().defined()) {
        cmp_kv_stride0 = cmp_kv.value().stride(0);
    }

    EXEC_NPU_CMD(aclnnKvQuantSparseAttnSharedkv, q, ori_kv, cmp_kv, ori_sparse_indices, cmp_sparse_indices,
                 ori_block_table, cmp_block_table, cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q,
                 seqused_kv, sinks, metadata, kv_quant_mode, tile_size, rope_head_dim, softmax_scale, cmp_ratio,
                 ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right, layout_q_ptr, layout_kv_ptr, ori_kv_stride0,
                 cmp_kv_stride0, return_softmax_lse, attn_out, softmax_lse);
    return std::tuple<at::Tensor, at::Tensor>(attn_out, softmax_lse);
}

auto get_valid_tensor = [](const c10::optional<at::Tensor> &tensor_opt, at::Device device) {
    return tensor_opt.has_value() ? tensor_opt : torch::empty({0}, torch::dtype(torch::kInt32).device(device));
};

at::Tensor kv_quant_sparse_attn_sharedkv_metadata(
    int64_t num_heads_q, int64_t num_heads_kv, int64_t head_dim, int64_t kv_quant_mode,
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_ori_kv,
    const c10::optional<at::Tensor> &cu_seqlens_cmp_kv, const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_kv, int64_t batch_size, int64_t max_seqlen_q, int64_t max_seqlen_kv,
    int64_t ori_topk, int64_t cmp_topk, int64_t tile_size, int64_t rope_head_dim, int64_t cmp_ratio,
    int64_t ori_mask_mode, int64_t cmp_mask_mode, int64_t ori_win_left, int64_t ori_win_right,
    c10::string_view layout_q, c10::string_view layout_kv, bool has_ori_kv, bool has_cmp_kv, c10::string_view device)
{
    constexpr int64_t OUTPUT_SIZE = 1024;
    at::Device output_device = at::Device(std::string(device));
    if (cu_seqlens_q.has_value()) {
        output_device = cu_seqlens_q.value().device();
    } else if (cu_seqlens_ori_kv.has_value()) {
        output_device = cu_seqlens_ori_kv.value().device();
    } else if (cu_seqlens_cmp_kv.has_value()) {
        output_device = cu_seqlens_cmp_kv.value().device();
    } else if (seqused_q.has_value()) {
        output_device = seqused_q.value().device();
    } else if (seqused_kv.has_value()) {
        output_device = seqused_kv.value().device();
    }
    at::Tensor output = torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(output_device));

    auto cu_seqlens_q_val = get_valid_tensor(cu_seqlens_q, output_device);
    auto cu_seqlens_ori_kv_val = get_valid_tensor(cu_seqlens_ori_kv, output_device);
    auto cu_seqlens_cmp_kv_val = get_valid_tensor(cu_seqlens_cmp_kv, output_device);
    auto seqused_q_val = get_valid_tensor(seqused_q, output_device);
    auto seqused_kv_val = get_valid_tensor(seqused_kv, output_device);

    std::string layout_q_str = std::string(layout_q);
    std::string layout_kv_str = std::string(layout_kv);
    char *layout_q_ptr = const_cast<char *>(layout_q_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    EXEC_NPU_CMD(aclnnKvQuantSparseAttnSharedkvMetadata, cu_seqlens_q_val, cu_seqlens_ori_kv_val, cu_seqlens_cmp_kv_val,
                 seqused_q_val, seqused_kv_val, num_heads_q, num_heads_kv, head_dim, batch_size, max_seqlen_q,
                 max_seqlen_kv, ori_topk, cmp_topk, kv_quant_mode, tile_size, rope_head_dim, cmp_ratio, ori_mask_mode,
                 cmp_mask_mode, ori_win_left, ori_win_right, layout_q_ptr, layout_kv_ptr, has_ori_kv, has_cmp_kv,
                 output);
    return output;
}

// --------------------------------------------------------------------- MoE

void check_hc_pre_shape_and_dtype(const at::Tensor &x, const at::Tensor &hc_fn, const at::Tensor &hc_scale,
                                  const at::Tensor &hc_base, int64_t hc_mult)
{
    constexpr int64_t HC_SCALE_SIZE = 3;
    auto x_dims = x.dim();
    TORCH_CHECK(x_dims == 3 || x_dims == 4, "Input tensor x's dim num should be 3 or 4, actual ", x_dims, ".");
    for (auto i = 0; i < x_dims; i++) {
        TORCH_CHECK(x.size(i) > 0, "Input tensor x's shape should be positive, but x.shape[", i, "] is ", x.size(i),
                    ".");
    }

    auto hc = x_dims == 4 ? x.size(2) : x.size(1);
    auto d = x_dims == 4 ? x.size(3) : x.size(2);
    TORCH_CHECK(hc == hc_mult, "The hc of x should be equal to hc_mult, actual hc is ", hc, ", hc_mult is ", hc_mult,
                ".");
    auto hc_mix = (2 + hc_mult) * hc_mult;
    TORCH_CHECK(hc_fn.dim() == 2, "Input tensor hc_fn's dim num should be 2, actual ", hc_fn.dim(), ".");
    TORCH_CHECK(hc_fn.size(0) == hc_mix, "The hc_fn.shape[0] should be (2 + hc_mult) * hc_mult, actual ", hc_fn.size(0),
                ", expected ", hc_mix, ".");
    TORCH_CHECK(hc_fn.size(1) == hc * d, "The hc_fn.shape[1] should be hc * d, actual hc_fn.shape[1] is ",
                hc_fn.size(1), ", hc is ", hc, ", d is ", d, ".");
    TORCH_CHECK(hc_scale.dim() == 1, "Input tensor hc_scale's dim num should be 1, actual ", hc_scale.dim(), ".");
    TORCH_CHECK(hc_scale.size(0) == HC_SCALE_SIZE, "Input tensor hc_scale's shape should be [", HC_SCALE_SIZE,
                "], actual [", hc_scale.size(0), "].");
    TORCH_CHECK(hc_base.dim() == 1, "Input tensor hc_base's dim num should be 1, actual ", hc_base.dim(), ".");
    TORCH_CHECK(hc_base.size(0) == hc_mix, "The hc_base.shape[0] should be (2 + hc_mult) * hc_mult, actual ",
                hc_base.size(0), ", expected ", hc_mix, ".");

    TORCH_CHECK(x.dtype() == at::kBFloat16, "x's dtype should be BFLOAT16.");
    TORCH_CHECK(hc_fn.dtype() == at::kFloat, "hc_fn's dtype should be FLOAT32.");
    TORCH_CHECK(hc_scale.dtype() == at::kFloat, "hc_scale's dtype should be FLOAT32.");
    TORCH_CHECK(hc_base.dtype() == at::kFloat, "hc_base's dtype should be FLOAT32.");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> construct_hc_pre_output_tensor(const at::Tensor &x, int64_t hc_mult)
{
    auto xDims = x.dim();
    at::SmallVector<int64_t, 8> y_size;
    at::SmallVector<int64_t, 8> post_size;
    at::SmallVector<int64_t, 8> comb_frag_size;
    if (xDims == 4) {
        auto batch = x.size(0);
        auto size = x.size(1);
        auto d = x.size(3);
        y_size = {batch, size, d};
        post_size = {batch, size, hc_mult};
        comb_frag_size = {batch, size, hc_mult, hc_mult};
    } else if (xDims == 3) {
        auto bs = x.size(0);
        auto d = x.size(2);
        y_size = {bs, d};
        post_size = {bs, hc_mult};
        comb_frag_size = {bs, hc_mult, hc_mult};
    }

    at::Tensor y = at::empty(y_size, x.options().dtype(at::kBFloat16));
    at::Tensor post = at::empty(post_size, x.options().dtype(at::kFloat));
    at::Tensor comb_frag = at::empty(comb_frag_size, x.options().dtype(at::kFloat));

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, post, comb_frag);
}

// Fused replacement for the hc_pre_inv_rms + at::linear + hc_pre_sinkhorn
// chain: the A5 kernel does the whole gating pre-pass on device, so unlike
// npu_hc_pre there is no intermediate `mixes` matmul on the torch side.
std::tuple<at::Tensor, at::Tensor, at::Tensor> hc_pre_v2(const at::Tensor &x, const at::Tensor &hc_fn,
                                                         const at::Tensor &hc_scale, const at::Tensor &hc_base,
                                                         int64_t hc_mult, int64_t hc_sinkhorn_iters, double norm_eps,
                                                         double hc_eps)
{
    check_hc_pre_shape_and_dtype(x, hc_fn, hc_scale, hc_base, hc_mult);

    auto output_tensors = construct_hc_pre_output_tensor(x, hc_mult);
    at::Tensor y = std::get<0>(output_tensors);
    at::Tensor post = std::get<1>(output_tensors);
    at::Tensor comb_frag = std::get<2>(output_tensors);
    EXEC_NPU_CMD(aclnnHcPre, x, hc_fn, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps, norm_eps, y, post,
                 comb_frag);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, post, comb_frag);
}

int64_t get_type_code(at::ScalarType dst_type)
{
    switch (dst_type) {
        case at::ScalarType::Float8_e5m2:
            return 35;
        case at::ScalarType::Float8_e4m3fn:
            return 36;
        case at::ScalarType::Half:
            return 1;
        case at::ScalarType::BFloat16:
            return 27;
        default:
            TORCH_CHECK(false, "Unsupported dtype: ", dst_type);
    }
    return 0;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
construct_swiglu_group_quant_output_tensor(const at::Tensor &x, int64_t dst_type, int64_t quant_mode, bool ue8m0_scale)
{
    constexpr int64_t SIZE = 8;
    constexpr int64_t SWIGLU_FACTOR = 2;
    constexpr int64_t PER_BLOCK_FP16 = 128;
    constexpr int64_t PER_MX_FP16 = 32;
    constexpr int64_t MX_SCALE_ALIGN_FACTOR = 2;
    constexpr int64_t GROUP_QUANT = 1;
    constexpr int64_t MX_QUANT = 2;
    constexpr int64_t FP8_QUANT = 3;

    at::SmallVector<int64_t, SIZE> y_size(x.sizes().begin(), x.sizes().end());
    for (size_t i = 0; i < x.sizes().size(); i++) {
        TORCH_CHECK(x.size(i) >= 0, "All values within x's shape should be non-negative, but shape[", i, "] is ",
                    x.size(i));
    }
    TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kBFloat16, "x should be FLOAT16 or BFLOAT16.");
    int64_t x_last_dim = x.sizes().back();
    TORCH_CHECK(quant_mode == GROUP_QUANT || quant_mode == MX_QUANT || quant_mode == FP8_QUANT,
                "Unsupported quant mode, only support ", GROUP_QUANT, " or ", MX_QUANT, " or ", FP8_QUANT, ".");
    if (quant_mode == GROUP_QUANT || quant_mode == FP8_QUANT) {
        TORCH_CHECK(x_last_dim % 256 == 0, "In group quant, the last dim of x should be divisible by 256, actual ",
                    x_last_dim, ".");
    } else {
        TORCH_CHECK(x_last_dim % 128 == 0, "In mx quant, the last dim of x should be divisible by 128, actual ",
                    x_last_dim, ".");
    }

    y_size.back() = y_size.back() / SWIGLU_FACTOR;
    int64_t y_last_dim = y_size.back();
    auto y_dtype = dst_type == 35 ? at::kFloat8_e5m2 : at::kFloat8_e4m3fn;
    at::Tensor y = at::empty(y_size, x.options().dtype(y_dtype));

    at::SmallVector<int64_t, SIZE> scale_size(y_size.begin(), y_size.end());
    if (quant_mode == GROUP_QUANT || quant_mode == FP8_QUANT) {
        scale_size.back() = (y_last_dim + PER_BLOCK_FP16 - 1) / PER_BLOCK_FP16;
    } else if (quant_mode == MX_QUANT) {
        int64_t scale_last_dim = (y_last_dim + PER_MX_FP16 - 1) / PER_MX_FP16;
        scale_last_dim = (scale_last_dim + MX_SCALE_ALIGN_FACTOR - 1) / MX_SCALE_ALIGN_FACTOR;
        scale_size.back() = scale_last_dim;
        scale_size.push_back(MX_SCALE_ALIGN_FACTOR);
    }

    auto scale_type = at::kFloat;
    if (quant_mode == MX_QUANT || (quant_mode == FP8_QUANT && ue8m0_scale)) {
        scale_type = at::kFloat8_e8m0fnu;
    }
    at::Tensor scale = at::empty(scale_size, x.options().dtype(scale_type));
    at::Tensor y_origin = at::empty(y_size, x.options().dtype(x.dtype()));

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, scale, y_origin);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> swiglu_group_quant(
    const at::Tensor &x, const c10::optional<at::Tensor> &topk_weight, const c10::optional<at::Tensor> &group_index,
    at::ScalarType dst_type, int64_t quant_mode, int64_t group_size, bool round_scale, bool ue8m0_scale,
    bool output_origin, int64_t group_list_type, double clamp_value)
{
    int64_t dst_type_code = get_type_code(dst_type);
    auto output_tensors = construct_swiglu_group_quant_output_tensor(x, dst_type_code, quant_mode, ue8m0_scale);
    at::Tensor y = std::get<0>(output_tensors);
    at::Tensor scale = std::get<1>(output_tensors);
    at::Tensor y_origin = std::get<2>(output_tensors);

    EXEC_NPU_CMD(aclnnSwigluGroupQuant, x, topk_weight, group_index, dst_type_code, quant_mode, group_size, round_scale,
                 ue8m0_scale, output_origin, group_list_type, clamp_value, y, scale, y_origin);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(y, scale, y_origin);
}

// ------------------------------------------------------------------ meta
// Fake-tensor implementations, so these ops can be traced under torch.compile
// and graph mode instead of falling back to eager. They only need to produce
// correctly shaped and typed outputs, and reuse the same output-construction
// helpers as the device path so the two cannot drift apart. Ports of
// vllm-ascend's csrc/torch_binding_meta.cpp.

namespace meta {

void indexer_compress_epilog(at::Tensor &indexer_compress_cache, at::Tensor &indexer_compress_cache_scale,
                             const at::Tensor &x, const at::Tensor &slot_mapping, int64_t quant_mode,
                             bool round_scale)
{
    return;
}

void indexer_compress_epilog_v2(at::Tensor &indexer_compress_cache, const at::Tensor &x,
                                const at::Tensor &slot_mapping, int64_t layout)
{
    return;
}

void kv_compress_epilog(at::Tensor &kv_compress_cache, const at::Tensor &x, const at::Tensor &slot_mapping,
                        int64_t quant_group_size, int64_t quant_mode, bool round_scale_flag, int64_t layout)
{
    return;
}

std::tuple<at::Tensor, at::Tensor> load_index_kv_cache(const at::Tensor &kv_cache, const at::Tensor &slot_mapping)
{
    return construct_load_index_kv_cache_output_tensor(kv_cache, slot_mapping);
}

std::tuple<at::Tensor, at::Tensor> kv_quant_sparse_attn_sharedkv(
    const at::Tensor &q, int64_t kv_quant_mode, const c10::optional<at::Tensor> &ori_kv,
    const c10::optional<at::Tensor> &cmp_kv, const c10::optional<at::Tensor> &ori_sparse_indices,
    const c10::optional<at::Tensor> &cmp_sparse_indices, const c10::optional<at::Tensor> &ori_block_table,
    const c10::optional<at::Tensor> &cmp_block_table, const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_ori_kv, const c10::optional<at::Tensor> &cu_seqlens_cmp_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &sinks, const c10::optional<at::Tensor> &metadata, int64_t tile_size,
    int64_t rope_head_dim, double softmax_scale, int64_t cmp_ratio, int64_t ori_mask_mode, int64_t cmp_mask_mode,
    int64_t ori_win_left, int64_t ori_win_right, c10::string_view layout_q, c10::string_view layout_kv,
    bool return_softmax_lse)
{
    return construct_sparse_attn_output_tensor(q, return_softmax_lse);
}

at::Tensor kv_quant_sparse_attn_sharedkv_metadata(
    int64_t num_heads_q, int64_t num_heads_kv, int64_t head_dim, int64_t kv_quant_mode,
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_ori_kv,
    const c10::optional<at::Tensor> &cu_seqlens_cmp_kv, const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_kv, int64_t batch_size, int64_t max_seqlen_q, int64_t max_seqlen_kv,
    int64_t ori_topk, int64_t cmp_topk, int64_t tile_size, int64_t rope_head_dim, int64_t cmp_ratio,
    int64_t ori_mask_mode, int64_t cmp_mask_mode, int64_t ori_win_left, int64_t ori_win_right,
    c10::string_view layout_q, c10::string_view layout_kv, bool has_ori_kv, bool has_cmp_kv,
    c10::string_view device)
{
    constexpr int64_t OUTPUT_SIZE = 1024;
    for (const auto &tensor : {cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q, seqused_kv}) {
        if (tensor.has_value()) {
            return torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(tensor.value().device()));
        }
    }

    // No input tensor to borrow a device from: honour the requested device
    // index but keep the tensor itself on meta.
    auto device_ori = at::Device(std::string(device));
    std::string device_str = "meta";
    if (device_ori.has_index()) {
        device_str += ":";
        device_str += std::to_string(device_ori.index());
    }
    return torch::empty({OUTPUT_SIZE}, torch::dtype(torch::kInt32).device(at::Device(device_str)));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> hc_pre_v2(const at::Tensor &x, const at::Tensor &hc_fn,
                                                         const at::Tensor &hc_scale, const at::Tensor &hc_base,
                                                         int64_t hc_mult, int64_t hc_sinkhorn_iters,
                                                         double norm_eps, double hc_eps)
{
    return construct_hc_pre_output_tensor(x, hc_mult);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> swiglu_group_quant(
    const at::Tensor &x, const c10::optional<at::Tensor> &topk_weight, const c10::optional<at::Tensor> &group_index,
    at::ScalarType dst_type, int64_t quant_mode, int64_t group_size, bool round_scale, bool ue8m0_scale,
    bool output_origin, int64_t group_list_type, double clamp_value)
{
    int64_t dst_type_code = get_type_code(dst_type);
    return construct_swiglu_group_quant_output_tensor(x, dst_type_code, quant_mode, ue8m0_scale);
}

}  // namespace meta

}  // namespace a5_ops
}  // namespace sglang

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def(
        "indexer_compress_epilog(Tensor(a!) indexer_compress_cache, Tensor(b!) indexer_compress_cache_scale, "
        "Tensor x, Tensor slot_mapping, int quant_mode=1, bool round_scale=True) -> ()");

    m.def(
        "indexer_compress_epilog_v2(Tensor(a!) indexer_compress_cache, Tensor x, Tensor slot_mapping, "
        "int layout=2) -> ()");

    m.def(
        "kv_compress_epilog(Tensor(a!) kv_compress_cache, Tensor x, Tensor slot_mapping, int quant_group_size, "
        "int quant_mode, bool round_scale_flag, int layout) -> ()");

    m.def("load_index_kv_cache(Tensor kv_cache, Tensor slot_mapping) -> (Tensor out, Tensor out_scale)");

    m.def(
        "kv_quant_sparse_attn_sharedkv(Tensor q, int kv_quant_mode, Tensor? ori_kv=None, Tensor? cmp_kv=None, "
        "Tensor? ori_sparse_indices=None, Tensor? cmp_sparse_indices=None, Tensor? ori_block_table=None, "
        "Tensor? cmp_block_table=None, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, "
        "Tensor? cu_seqlens_cmp_kv=None, Tensor? seqused_q=None, Tensor? seqused_kv=None, Tensor? sinks=None, "
        "Tensor? metadata=None, int tile_size=0, int rope_head_dim=0, float softmax_scale=0.0, int cmp_ratio=0, "
        "int ori_mask_mode=4, int cmp_mask_mode=3, int ori_win_left=127, int ori_win_right=0, "
        "str layout_q='BSND', str layout_kv='PA_ND', bool return_softmax_lse=False) "
        "-> (Tensor out, Tensor softmax_lse)");

    m.def(
        "kv_quant_sparse_attn_sharedkv_metadata(int num_heads_q, int num_heads_kv, int head_dim, "
        "int kv_quant_mode, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, "
        "Tensor? cu_seqlens_cmp_kv=None, Tensor? seqused_q=None, Tensor? seqused_kv=None, int batch_size=0, "
        "int max_seqlen_q=0, int max_seqlen_kv=0, int ori_topk=0, int cmp_topk=0, int tile_size=0, "
        "int rope_head_dim=0, int cmp_ratio=-1, int ori_mask_mode=4, int cmp_mask_mode=3, int ori_win_left=127, "
        "int ori_win_right=0, str layout_q='BSND', str layout_kv='PA_ND', bool has_ori_kv=True, "
        "bool has_cmp_kv=True, str device='npu') -> Tensor");

    m.def(
        "hc_pre_v2(Tensor x, Tensor hc_fn, Tensor hc_scale, Tensor hc_base, int hc_mult, int hc_sinkhorn_iters, "
        "float norm_eps, float hc_eps) -> (Tensor out0, Tensor out1, Tensor out2)");

    m.def(
        "swiglu_group_quant(Tensor x, Tensor? topk_weight, Tensor? group_index, ScalarType dst_type=39, "
        "int quant_mode=1, int group_size=128, bool round_scale=False, bool ue8m0_scale=False, "
        "bool output_origin=False, int group_list_type=0, float clamp_value=0.0) "
        "-> (Tensor y, Tensor scale, Tensor y_origin)");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("indexer_compress_epilog", TORCH_FN(sglang::a5_ops::indexer_compress_epilog));
    m.impl("indexer_compress_epilog_v2", TORCH_FN(sglang::a5_ops::indexer_compress_epilog_v2));
    m.impl("kv_compress_epilog", TORCH_FN(sglang::a5_ops::kv_compress_epilog));
    m.impl("load_index_kv_cache", TORCH_FN(sglang::a5_ops::load_index_kv_cache));
    m.impl("kv_quant_sparse_attn_sharedkv", TORCH_FN(sglang::a5_ops::kv_quant_sparse_attn_sharedkv));
    m.impl("kv_quant_sparse_attn_sharedkv_metadata", TORCH_FN(sglang::a5_ops::kv_quant_sparse_attn_sharedkv_metadata));
    m.impl("hc_pre_v2", TORCH_FN(sglang::a5_ops::hc_pre_v2));
    m.impl("swiglu_group_quant", TORCH_FN(sglang::a5_ops::swiglu_group_quant));
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("indexer_compress_epilog", TORCH_FN(sglang::a5_ops::meta::indexer_compress_epilog));
    m.impl("indexer_compress_epilog_v2", TORCH_FN(sglang::a5_ops::meta::indexer_compress_epilog_v2));
    m.impl("kv_compress_epilog", TORCH_FN(sglang::a5_ops::meta::kv_compress_epilog));
    m.impl("load_index_kv_cache", TORCH_FN(sglang::a5_ops::meta::load_index_kv_cache));
    m.impl("kv_quant_sparse_attn_sharedkv", TORCH_FN(sglang::a5_ops::meta::kv_quant_sparse_attn_sharedkv));
    m.impl("kv_quant_sparse_attn_sharedkv_metadata",
           TORCH_FN(sglang::a5_ops::meta::kv_quant_sparse_attn_sharedkv_metadata));
    m.impl("hc_pre_v2", TORCH_FN(sglang::a5_ops::meta::hc_pre_v2));
    m.impl("swiglu_group_quant", TORCH_FN(sglang::a5_ops::meta::swiglu_group_quant));
}

}  // namespace
