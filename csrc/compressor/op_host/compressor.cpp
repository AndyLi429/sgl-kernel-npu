#include <ATen/ATen.h>
#include <ATen/Context.h>

#include <algorithm>
#include <vector>

#include "acl/acl.h"
#include "aclrtlaunch_compressor.h"
#include "defines.h"
#include "ge_helper.h"
#include "tiling/compressor_tiling.h"
#include "torch_helper.h"

namespace sglang {
namespace npu_kernel {
namespace {

constexpr int64_t MIN_HIDDEN_SIZE = 1024;
constexpr int64_t MAX_HIDDEN_SIZE = 10240;
constexpr int64_t HIDDEN_SIZE_ALIGNMENT = 512;
constexpr int64_t MIN_CMP_RATIO = 2;
constexpr int64_t MAX_CMP_RATIO = 128;
constexpr int64_t MAX_BLOCK_SIZE = 1024;

using namespace ge_helper;

class CompressorOpDef : public OpDef
{
public:
    explicit CompressorOpDef(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("wkv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("wgate")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("state_cache").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        this->Input("ape")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("state_block_table")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cu_seqlens")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("seqused")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("start_pos")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("cmp_kv").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND});
        this->Output("state_cache").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        this->Output("softmax_score").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        this->Output("kv").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        this->Attr("cmp_ratio").AttrType(REQUIRED).Int(4);
        this->Attr("coff").AttrType(OPTIONAL).Int(1);
        this->Attr("cache_mode").AttrType(OPTIONAL).Int(1);
        this->Attr("state_cache_stride_dim0").AttrType(OPTIONAL).Int(0);
        this->Attr("grad_enabled").AttrType(OPTIONAL).Bool(false);
    }
};

void CheckTensorMetadata(const at::Tensor &tensor, const at::Tensor &x, const char *name, bool requireContiguous)
{
    TORCH_CHECK(tensor.device() == x.device(), name, " must be on the same NPU device as x");
    TORCH_CHECK(tensor.layout() == at::kStrided, name, " must have strided layout");
    TORCH_CHECK(!requireContiguous || tensor.is_contiguous(), name, " must be contiguous");
}

void CheckOptionalIntTensor(const c10::optional<at::Tensor> &tensor, const at::Tensor &x, const char *name,
                            int64_t batchSize)
{
    if (!tensor.has_value()) {
        return;
    }
    CheckTensorMetadata(*tensor, x, name, true);
    TORCH_CHECK(tensor->scalar_type() == at::kInt, name, " must be INT32");
    TORCH_CHECK(tensor->dim() == 1, name, " must be 1D");
    TORCH_CHECK(tensor->size(0) == batchSize, name, " size must equal batch size");
}

void ValidateCompressorInputs(const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &wgate,
                              const at::Tensor &state_cache, const at::Tensor &ape,
                              const c10::optional<at::Tensor> &stateBlockTable,
                              const c10::optional<at::Tensor> &cuSeqlens, const c10::optional<at::Tensor> &seqused,
                              const c10::optional<at::Tensor> &startPos, int64_t cmpRatio, int64_t coff,
                              int64_t cacheMode)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be an NPU tensor");
    CheckTensorMetadata(x, x, "x", true);
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16, "x must be FP16 or BF16");
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "x must use TH [T, H] or BSH [B, S, H] layout");

    TORCH_CHECK(cmpRatio >= MIN_CMP_RATIO && cmpRatio <= MAX_CMP_RATIO, "cmp_ratio must be within [2, 128]");
    TORCH_CHECK(coff == 1 || coff == 2, "coff must be 1 or 2");
    TORCH_CHECK(cacheMode == 1 || cacheMode == 2, "cache_mode must be 1 (Linear) or 2 (Ring)");

    const int64_t hiddenSize = x.size(x.dim() - 1);
    TORCH_CHECK(
        hiddenSize >= MIN_HIDDEN_SIZE && hiddenSize <= MAX_HIDDEN_SIZE && hiddenSize % HIDDEN_SIZE_ALIGNMENT == 0,
        "x hidden size must be within [1024, 10240] and 512-aligned");

    CheckTensorMetadata(wkv, x, "wkv", true);
    CheckTensorMetadata(wgate, x, "wgate", true);
    TORCH_CHECK(wkv.scalar_type() == x.scalar_type(), "wkv dtype must match x");
    TORCH_CHECK(wgate.scalar_type() == x.scalar_type(), "wgate dtype must match x");
    TORCH_CHECK(wkv.dim() == 2 && wgate.dim() == 2, "wkv and wgate must be 2D");
    TORCH_CHECK(wkv.sizes() == wgate.sizes(), "wkv and wgate shapes must match");
    TORCH_CHECK(wkv.size(1) == hiddenSize, "wkv and wgate dim 1 must equal x hidden size");
    TORCH_CHECK(wkv.size(0) > 0 && wkv.size(0) % coff == 0, "wkv dim 0 must equal coff*D");
    const int64_t headDim = wkv.size(0) / coff;
    TORCH_CHECK(headDim == 128 || headDim == 512, "D must be 128 or 512");

    CheckTensorMetadata(state_cache, x, "state_cache", false);
    TORCH_CHECK(state_cache.scalar_type() == at::kFloat, "state_cache must be FP32");
    TORCH_CHECK(state_cache.dim() == 3 && state_cache.stride(0) > 0,
                "state_cache must be [block_num, block_size, 2*coff*D] with positive dim-0 stride");
    TORCH_CHECK(state_cache.size(0) > 0 && state_cache.size(1) > 0 && state_cache.size(1) <= MAX_BLOCK_SIZE &&
                    state_cache.size(2) == 2 * coff * headDim,
                "state_cache shape must be [block_num, block_size, 2*coff*D] with block_size in [1, 1024]");
    TORCH_CHECK(state_cache.stride(2) == 1 && state_cache.stride(1) == state_cache.size(2),
                "state_cache inner dimensions must be contiguous");

    CheckTensorMetadata(ape, x, "ape", true);
    TORCH_CHECK(ape.scalar_type() == at::kFloat, "ape must be FP32");
    TORCH_CHECK(ape.dim() == 2 && ape.size(0) == cmpRatio && ape.size(1) == coff * headDim,
                "ape shape must be [cmp_ratio, coff*D]");

    TORCH_CHECK(stateBlockTable.has_value(), "state_block_table is required");
    CheckTensorMetadata(*stateBlockTable, x, "state_block_table", true);
    TORCH_CHECK(stateBlockTable->scalar_type() == at::kInt, "state_block_table must be INT32");
    const int64_t expectedTableDim = cacheMode == 2 ? 1 : 2;
    TORCH_CHECK(stateBlockTable->dim() == expectedTableDim, "state_block_table must be ", expectedTableDim,
                "D for cache_mode ", cacheMode);

    int64_t batchSize = 0;
    if (x.dim() == 3) {
        TORCH_CHECK(!cuSeqlens.has_value(), "cu_seqlens must be absent for BSH layout");
        batchSize = x.size(0);
    } else {
        TORCH_CHECK(cuSeqlens.has_value(), "cu_seqlens is required for TH layout");
        CheckTensorMetadata(*cuSeqlens, x, "cu_seqlens", true);
        TORCH_CHECK(cuSeqlens->scalar_type() == at::kInt, "cu_seqlens must be INT32");
        TORCH_CHECK(cuSeqlens->dim() == 1 && cuSeqlens->size(0) >= 1, "cu_seqlens must be [B+1]");
        batchSize = cuSeqlens->size(0) - 1;
    }
    TORCH_CHECK(stateBlockTable->size(0) == batchSize, "state_block_table dim 0 must equal batch size");
    TORCH_CHECK(cacheMode != 2 || state_cache.size(0) >= batchSize,
                "state_cache block_num must not be less than batch size in Ring mode");

    CheckOptionalIntTensor(seqused, x, "seqused", batchSize);
    CheckOptionalIntTensor(startPos, x, "start_pos", batchSize);
}

struct CompressorOutputs {
    at::Tensor cmpKv;
    at::Tensor stateCache;
    at::Tensor softmaxScore;
    at::Tensor kv;
};

CompressorOutputs CreateCompressorOutputs(const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &stateCache,
                                          const c10::optional<at::Tensor> &cuSeqlens, int64_t cmpRatio, int64_t coff)
{
    const int64_t headDim = wkv.size(0) / coff;
    int64_t compressedTokens = 0;
    std::vector<int64_t> cmpKvShape;
    std::vector<int64_t> privateOutputShape;
    if (x.dim() == 3) {
        compressedTokens = (x.size(1) + cmpRatio - 1) / cmpRatio;
        cmpKvShape = {x.size(0), compressedTokens, headDim};
        privateOutputShape = {x.size(0), compressedTokens, coff * cmpRatio, headDim};
    } else {
        const int64_t batchSize = cuSeqlens->size(0) - 1;
        compressedTokens = std::min(x.size(0), x.size(0) / cmpRatio + batchSize);
        cmpKvShape = {compressedTokens, headDim};
        privateOutputShape = {compressedTokens, coff * cmpRatio, headDim};
    }

    auto xOptions = x.options().requires_grad(false);
    auto floatOptions = xOptions.dtype(at::kFloat);
    return {at::empty(cmpKvShape, xOptions), at::empty(stateCache.sizes(), floatOptions),
            at::empty(privateOutputShape, floatOptions), at::empty(privateOutputShape, floatOptions)};
}

void *OptionalTensorData(const c10::optional<at::Tensor> &tensor)
{
    return tensor.has_value() ? tensor->data_ptr() : nullptr;
}

}  // namespace

HOST_API at::Tensor compressor(const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &wgate,
                               at::Tensor &state_cache, const at::Tensor &ape,
                               c10::optional<at::Tensor> state_block_table, c10::optional<at::Tensor> cu_seqlens,
                               c10::optional<at::Tensor> seqused, c10::optional<at::Tensor> start_pos,
                               int64_t cmp_ratio, int64_t coff, int64_t cache_mode)
{
    ValidateCompressorInputs(x, wkv, wgate, state_cache, ape, state_block_table, cu_seqlens, seqused, start_pos,
                             cmp_ratio, coff, cache_mode);

    CompressorOutputs outputs = CreateCompressorOutputs(x, wkv, state_cache, cu_seqlens, cmp_ratio, coff);
    const auto state_cache_stride_dim0 = state_cache.stride(0);
    const auto deterministic_level = at::globalContext().deterministicAlgorithms() ? 3 : 0;

    CompressorOpDef compressorDef("compressor");
    compressorDef.SetAttrAny("cmp_ratio", static_cast<int>(cmp_ratio));
    compressorDef.SetAttrAny("coff", static_cast<int>(coff));
    compressorDef.SetAttrAny("cache_mode", static_cast<int>(cache_mode));
    compressorDef.SetAttrAny("state_cache_stride_dim0", state_cache_stride_dim0);
    compressorDef.SetAttrAny("grad_enabled", false);

    auto context = std::make_shared<TilingContext>("compressor");
    auto scalarType = x.scalar_type();
    compressorDef.SetToContext(context, scalarType);
    context->SetDeterministicLevel(deterministic_level);
    context->SetWorkspaceSizes(0);
    context->RegisterTensor(x, true);
    context->RegisterTensor(wkv, true);
    context->RegisterTensor(wgate, true);
    context->RegisterTensor(state_cache, true);
    context->RegisterTensor(ape, true);
    context->RegisterTensor(state_block_table, true);
    context->RegisterTensor(cu_seqlens, true);
    context->RegisterTensor(seqused, true);
    context->RegisterTensor(start_pos, true);
    context->RegisterTensor(outputs.cmpKv, false);
    context->RegisterTensor(outputs.stateCache, false);
    context->RegisterTensor(outputs.softmaxScore, false);
    context->RegisterTensor(outputs.kv, false);

    optiling::CompressorTilingData tilingData{};
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    TORCH_CHECK(optiling::TilingCompressorArch35(context.get(), &tilingData, tilingKey, blockDim) == ge::GRAPH_SUCCESS,
                "compressor tiling failed");
    (void)tilingKey;

    const auto tilingSize = static_cast<int64_t>(sizeof(tilingData));
    auto byteOptions = x.options().dtype(at::kByte).requires_grad(false);
    at::Tensor tiling = at::empty({tilingSize}, byteOptions);
    const auto copyStatus =
        aclrtMemcpy(tiling.data_ptr(), tilingSize, &tilingData, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE);
    TORCH_CHECK(copyStatus == ACL_SUCCESS, "compressor tiling H2D copy failed, ACL error: ", copyStatus);

    const auto workspaceSize = static_cast<int64_t>(*context->GetWorkspaceSizes(1));
    at::Tensor workspace = at::empty({workspaceSize}, byteOptions);
    void *stateBlockTableData = OptionalTensorData(state_block_table);
    void *cuSeqlensData = OptionalTensorData(cu_seqlens);
    void *sequsedData = OptionalTensorData(seqused);
    void *startPosData = OptionalTensorData(start_pos);
    EXEC_KERNEL_CMD(compressor, blockDim, x, wkv, wgate, state_cache, ape, stateBlockTableData, cuSeqlensData,
                    sequsedData, startPosData, outputs.cmpKv, outputs.stateCache, outputs.softmaxScore, outputs.kv,
                    workspace, tiling);
    return outputs.cmpKv;
}

}  // namespace npu_kernel
}  // namespace sglang
