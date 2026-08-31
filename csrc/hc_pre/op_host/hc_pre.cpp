#include "pytorch_npu_helper.h"
#include "sgl_kenel_npu_ops.h"

namespace sglang::npu_kernel {
namespace {
constexpr std::string_view HC_PRE_OP_NAME = "aclnnHcPre";
constexpr int64_t HC_MULT = 4;
constexpr int64_t MIX_HC = 24;

void check_inputs(const at::Tensor &x, const at::Tensor &hc_fn,
                  const at::Tensor &hc_scale, const at::Tensor &hc_base,
                  int64_t hc_mult)
{
    TORCH_CHECK(x.dim() == 3 || x.dim() == 4, "x must be rank 3 or 4");
    const auto hc_dim = x.dim() == 4 ? 2 : 1;
    const auto d_dim = x.dim() == 4 ? 3 : 2;
    TORCH_CHECK(hc_mult == HC_MULT && x.size(hc_dim) == HC_MULT,
                "hc_mult must be 4");
    TORCH_CHECK(x.size(d_dim) == 4096 || x.size(d_dim) == 7168,
                "hidden size must be 4096 or 7168");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be BF16");
    TORCH_CHECK(hc_fn.dim() == 2 && hc_fn.size(0) == MIX_HC &&
                    hc_fn.size(1) == HC_MULT * x.size(d_dim),
                "hc_fn must have shape [24, 4 * hidden_size]");
    TORCH_CHECK(hc_scale.dim() == 1 && hc_scale.size(0) == 3,
                "hc_scale must have shape [3]");
    TORCH_CHECK(hc_base.dim() == 1 && hc_base.size(0) == MIX_HC,
                "hc_base must have shape [24]");
    TORCH_CHECK(hc_fn.scalar_type() == at::kFloat && hc_scale.scalar_type() == at::kFloat &&
                    hc_base.scalar_type() == at::kFloat,
                "hc_fn, hc_scale, and hc_base must be FP32");
}
}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_hc_pre_v2(
    const at::Tensor &x, const at::Tensor &hc_fn, const at::Tensor &hc_scale,
    const at::Tensor &hc_base, int64_t hc_mult, int64_t hc_sinkhorn_iters,
    double norm_eps, double hc_eps)
{
    check_inputs(x, hc_fn, hc_scale, hc_base, hc_mult);
    const auto y_shape = x.dim() == 4 ? at::IntArrayRef({x.size(0), x.size(1), x.size(3)})
                                      : at::IntArrayRef({x.size(0), x.size(2)});
    const auto post_shape = x.dim() == 4 ? at::IntArrayRef({x.size(0), x.size(1), hc_mult})
                                         : at::IntArrayRef({x.size(0), hc_mult});
    const auto comb_shape = x.dim() == 4 ? at::IntArrayRef({x.size(0), x.size(1), hc_mult, hc_mult})
                                         : at::IntArrayRef({x.size(0), hc_mult, hc_mult});
    auto y = at::empty(y_shape, x.options());
    auto post = at::empty(post_shape, x.options().dtype(at::kFloat));
    auto comb = at::empty(comb_shape, x.options().dtype(at::kFloat));
    EXEC_NPU_CMD<HC_PRE_OP_NAME>(x, hc_fn, hc_scale, hc_base, hc_mult,
                                 hc_sinkhorn_iters, hc_eps, norm_eps, y, post, comb);
    return {y, post, comb};
}
}  // namespace sglang::npu_kernel
