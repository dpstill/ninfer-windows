// Implements: include/ninfer/ops/log_softmax.h
// Match: wrapper-validated contiguous tensors and valid vocabulary rows.
// Algorithm assumptions: one independent CTA per column; no global workspace.
#include "ops/launcher/log_softmax.h"

#include "core/device.h"
#include "ops/kernel/log_softmax.cuh"

namespace ninfer::ops::detail {

void log_softmax_launch(const Tensor& logits, std::int32_t valid_rows, Tensor& output,
                        cudaStream_t stream) {
    const auto columns = static_cast<unsigned int>(logits.ne[1]);
    log_softmax_kernel<kLogSoftmaxBlock><<<columns, kLogSoftmaxBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), static_cast<float*>(output.data),
        valid_rows, logits.ne[0]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
