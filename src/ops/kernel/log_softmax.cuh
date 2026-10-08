#pragma once

// Implements: include/ninfer/ops/log_softmax.h
// Match: contiguous BF16 [physical_rows,C] and FP32 [valid_rows,C].
// Algorithm assumptions: one 256-thread CTA performs a stable per-column log-softmax.

#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kLogSoftmaxBlock = 256;

template <int BlockSize>
__device__ __forceinline__ float log_softmax_block_max(float value) {
    static_assert(BlockSize >= kWarpSize && BlockSize <= 1024);
    static_assert((BlockSize & (BlockSize - 1)) == 0);
    constexpr int kWarps = BlockSize / kWarpSize;
    __shared__ float warp_maxima[kWarps];
    __shared__ float result;

    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    value          = warp_max(value);
    if (lane == 0) { warp_maxima[warp] = value; }
    __syncthreads();

    if (warp == 0) {
        value = lane < kWarps ? warp_maxima[lane] : -CUDART_INF_F;
        value = warp_max(value);
        if (lane == 0) { result = value; }
    }
    __syncthreads();
    return result;
}

template <int BlockSize>
__launch_bounds__(BlockSize) __global__
    void log_softmax_kernel(const __nv_bfloat16* logits, float* output, std::int32_t valid_rows,
                            std::int32_t physical_rows) {
    const std::int32_t column    = static_cast<std::int32_t>(blockIdx.x);
    const std::int64_t base      = static_cast<std::int64_t>(column) * physical_rows;
    const std::int64_t out_base  = static_cast<std::int64_t>(column) * valid_rows;

    float local_max = -CUDART_INF_F;
    for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
         row += BlockSize) {
        local_max = fmaxf(local_max, __bfloat162float(logits[base + row]));
    }
    const float maximum = log_softmax_block_max<BlockSize>(local_max);

    float local_sum = 0.0f;
    for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
         row += BlockSize) {
        local_sum += expf(__bfloat162float(logits[base + row]) - maximum);
    }
    __shared__ float warp_sums[BlockSize / kWarpSize];
    const float block_sum = block_reduce_sum<BlockSize>(local_sum, warp_sums);
    __shared__ float log_denominator;
    if (threadIdx.x == 0) { log_denominator = maximum + logf(block_sum); }
    __syncthreads();

    for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
         row += BlockSize) {
        output[out_base + row] = __bfloat162float(logits[base + row]) - log_denominator;
    }
}

} // namespace ninfer::ops
