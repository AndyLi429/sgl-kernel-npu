#!/bin/bash
# Build the Ascend950 (A5) custom operator package.
#
# The operator sources and the cann-ops-transformer build framework under this
# directory are vendored from vllm-ascend (PR #9271, merge 55edd9f). Keeping the
# vendored framework rather than re-expressing the ops in the direct-launch style
# used by csrc/<op>/ is deliberate: the per-op op_host/CMakeLists.txt carry
# compile options (--cce-auto-sync=off, --op_relocatable_kernel_binary=true,
# -mllvm -cce-aicore-hoist-movemask=false) that the generated kernel binaries
# depend on for performance parity with vllm-ascend.
#
# Output: a custom OPP run package installed under
#   python/sgl_kernel_npu/sgl_kernel_npu/vendors/
# which sgl_kernel_npu/__init__.py exposes via ASCEND_CUSTOM_OPP_PATH.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SOC_ARG="${SOC_ARG:-ascend950}"
VENDOR_NAME="${VENDOR_NAME:-sglang}"
INSTALL_DIR="${INSTALL_DIR:-${REPO_ROOT}/python/sgl_kernel_npu/sgl_kernel_npu/vendors}"

# The A5 operator set from vllm-ascend PR #9271. Ops that already existed on
# vllm-ascend main (compressor, quant_lightning_indexer{,_metadata},
# moe_gating_top_k_hash, inplace_partial_rotary_mul, hc_pre_sinkhorn,
# hc_pre_inv_rms, hc_post) are NOT vendored here — see csrc/a5_ops/README.md.
A5_OPS_ARRAY=(
    "indexer_compress_epilog"
    "indexer_compress_epilog_v2"
    "kv_compress_epilog"
    "kv_quant_sparse_attn_sharedkv"
    "kv_quant_sparse_attn_sharedkv_metadata"
    "load_index_kv_cache"
    "hc_pre"
    "swiglu_group_quant"
)

log() { echo "[build_a5_ops] $*"; }

if [ -z "${ASCEND_HOME_PATH}" ]; then
    echo "Error: ASCEND_HOME_PATH is unset. Source set_env.sh first." >&2
    exit 1
fi
log "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"

# Every op directory in the list must resolve, otherwise the framework silently
# produces a package missing that operator and the failure only shows up as a
# runtime "op not found" much later.
for op_name in "${A5_OPS_ARRAY[@]}"; do
    op_dir=""
    for candidate in "${SCRIPT_DIR}/attention/${op_name}" "${SCRIPT_DIR}/moe/${op_name}"; do
        if [ -d "${candidate}" ]; then
            op_dir="${candidate}"
            break
        fi
    done
    if [ -z "${op_dir}" ]; then
        echo "Error: operator directory for '${op_name}' not found under ${SCRIPT_DIR}" >&2
        exit 1
    fi
    log "op ${op_name}: dir=${op_dir}"
done

CUSTOM_OPS=$(IFS=';'; echo "${A5_OPS_ARRAY[*]}")

cd "${SCRIPT_DIR}"
log "cleaning previous build output"
rm -rf -- build output build_out

log "build command: bash build.sh --pkg --ops=\"${CUSTOM_OPS}\" --soc=\"${SOC_ARG}\" --vendor_name=\"${VENDOR_NAME}\""
bash build.sh --pkg --ops="${CUSTOM_OPS}" --soc="${SOC_ARG}" --vendor_name="${VENDOR_NAME}"

shopt -s nullglob
installers=(./build/cann-ops-transformer*.run)
shopt -u nullglob

if (( ${#installers[@]} != 1 )); then
    echo "Error: expected exactly 1 run package, got ${#installers[@]}" >&2
    exit 1
fi
log "run package: $(ls -lh "${installers[0]}")"

mkdir -p -- "${INSTALL_DIR}"
find "${INSTALL_DIR}" -mindepth 1 -maxdepth 1 ! -name '.gitkeep' -exec rm -rf -- {} +

chmod +x -- "${installers[0]}"
log "installing into ${INSTALL_DIR}"
"${installers[0]}" --install-path="${INSTALL_DIR}"

log "installed tree:"
find "${INSTALL_DIR}" -mindepth 1 -maxdepth 3 -print | sort | sed 's#^#[build_a5_ops] install: #'
log "done"
