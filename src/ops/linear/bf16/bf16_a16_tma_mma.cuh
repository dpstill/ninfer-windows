#pragma once

#include "ops/common/mbarrier.cuh"
#include "ops/linear/bf16/bf16_mma_common.cuh"
#include "ops/linear/bf16/bf16_operands.h"
#include "ops/common/token_slices.h"
#include "ops/common/math.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

struct alignas(128) Bf16TmaDescriptors {
    CUtensorMap weight;
    CUtensorMap activation;
};

#ifdef _WIN32
// MSVC cannot pass an alignas(128) struct by value as a kernel parameter (C2719). Keep the
// descriptor block in a device buffer and hand the kernel a pointer; the TMA unit reads the
// tensor map from that address. POSIX keeps the by-value __grid_constant__ spelling.
#define NINFER_BF16_TMA_DESCRIPTOR_PARAM const Bf16TmaDescriptors* __restrict__
#else
#define NINFER_BF16_TMA_DESCRIPTOR_PARAM const __grid_constant__ Bf16TmaDescriptors
#endif

inline CUtensorMap bf16_tma_map(const __nv_bfloat16* pointer, int rows, int k, int block_rows,
                                int block_k) {
    // Factor K into 64-element sectors. The contiguous 128-byte dimension matches the
    // hardware swizzle while the next box dimension permits larger K tiles without repacking.
    CUtensorMap result{};
    const std::uint64_t dimensions[]{64, static_cast<std::uint64_t>(k / 64),
                                     static_cast<std::uint64_t>(rows)};
    const std::uint64_t strides[]{128, static_cast<std::uint64_t>(k) * 2};
    const std::uint32_t box[]{64, static_cast<std::uint32_t>(block_k / 64),
                              static_cast<std::uint32_t>(block_rows)};
    const std::uint32_t steps[]{1, 1, 1};
    const auto status = cuTensorMapEncodeTiled(
        &result, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<__nv_bfloat16*>(pointer),
        dimensions, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(status, &name);
        throw std::runtime_error(std::string("BF16 TMA descriptor: ") +
                                 (name ? name : "CUDA error"));
    }
    return result;
}

template <class Schedule>
Bf16TmaDescriptors make_bf16_tma_descriptors(const Bf16A16Operands& p) {
    validate_bf16_operands<Schedule>(p);
    if (p.rows % Schedule::kBlockRows || p.k % Schedule::kBlockK)
        throw std::invalid_argument("BF16 TMA requires complete row/K tiles");
    return {bf16_tma_map(p.weight, p.rows, p.k, Schedule::kBlockRows, Schedule::kBlockK),
            bf16_tma_map(p.x, p.tokens, p.k, Schedule::kBlockTokens, Schedule::kBlockK)};
}

__device__ __forceinline__ void bf16_tma_load(void* destination, const CUtensorMap* map,
                                              int k_sector, int row, std::uint64_t* barrier) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4}], [%5];"
                 :
                 : "r"(smem_addr(destination)), "l"(map), "r"(0), "r"(k_sector), "r"(row),
                   "r"(smem_addr(barrier))
                 : "memory");
}

template <class Schedule, class Epilogue>
inline constexpr int bf16_tma_scratch_bytes =
    ((Schedule::kTensorBytes > bf16_epilogue_bytes<Schedule, Epilogue>
          ? Schedule::kTensorBytes
          : bf16_epilogue_bytes<Schedule, Epilogue>)+127) /
    128 * 128;

template <class Schedule, bool FullTokens, class Output, class Epilogue>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void bf16_a16_tma_mma_kernel(
    NINFER_BF16_TMA_DESCRIPTOR_PARAM descriptors, Output output, Epilogue epilogue,
    int rows, int input_rows, int token_offset, int count) {
    constexpr int BR = Schedule::kBlockRows, BT = Schedule::kBlockTokens, BK = Schedule::kBlockK;
    constexpr int S   = Schedule::kStages;
    const int K       = Schedule::kStaticK ? Schedule::kStaticK : input_rows;
    const int tiles_r = rows / BR, tiles_t = (count + BT - 1) / BT;
    int tile_r, tile_t;
    bf16_mma_tile_coordinates<Schedule>(blockIdx.x, tiles_r, tiles_t, tile_r, tile_t);
    const int row_begin = tile_r * BR, token_begin = token_offset + tile_t * BT;
    extern __shared__ __align__(128) unsigned char storage[];
    auto* a = reinterpret_cast<__nv_bfloat16*>(storage);
    auto* b = a + S * BR * BK;
    auto* full =
        reinterpret_cast<std::uint64_t*>(storage + bf16_tma_scratch_bytes<Schedule, Epilogue>);
    auto* empty = full + S;
    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < S; ++stage) {
            cta_mbarrier_init(full + stage, 1);
            cta_mbarrier_init(empty + stage, Schedule::kConsumerWarps);
        }
        cta_mbarrier_fence_init();
    }
    __syncthreads();
    const int tiles_k = K / BK;
    if (threadIdx.x < Schedule::kProducerThreads) {
        if (threadIdx.x == 0) {
#ifdef _WIN32
            const Bf16TmaDescriptors* descriptor_block = descriptors;
#else
            const Bf16TmaDescriptors* descriptor_block = &descriptors;
#endif
            for (int kt = 0; kt < tiles_k; ++kt) {
                const int stage = kt % S;
                cta_mbarrier_wait(empty + stage, 1U ^ ((kt / S) & 1U));
                cta_mbarrier_arrive_expect_tx(full + stage, (BR + BT) * BK * 2);
                bf16_tma_load(a + stage * BR * BK, &descriptor_block->weight, kt * (BK / 64), row_begin,
                              full + stage);
                bf16_tma_load(b + stage * BT * BK, &descriptor_block->activation, kt * (BK / 64),
                              token_begin, full + stage);
            }
        }
        return;
    }
    const int tid  = threadIdx.x - Schedule::kProducerThreads;
    const int warp = tid / 32, lane = tid & 31;
    float accum[Schedule::kMmaRows][Schedule::kMmaTokens][4] = {};
    for (int kt = 0; kt < tiles_k; ++kt) {
        const int stage = kt % S;
        cta_mbarrier_wait(full + stage, (kt / S) & 1U);
        bf16_mma_compute_stage<Schedule>(a + stage * BR * BK, b + stage * BT * BK, accum, warp,
                                         lane);
        // Every lane must finish its shared reads before the elected lane releases this stage.
        __syncwarp();
        if (lane == 0) cta_mbarrier_arrive(empty + stage);
    }
    bf16_finish_mma_tile<Schedule, FullTokens>(output, epilogue, storage, accum, row_begin,
                                               token_begin, rows, token_offset + count, warp, lane);
}

#ifdef _WIN32
// cudaMallocAsync is pool-backed; the free is stream-ordered after the kernel that
// consumes the block.
struct Bf16TmaDescriptorBlock {
    Bf16TmaDescriptors* device = nullptr;

    explicit Bf16TmaDescriptorBlock(cudaStream_t stream) {
        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&device),
                                   sizeof(Bf16TmaDescriptors), stream));
    }

    Bf16TmaDescriptorBlock(const Bf16TmaDescriptorBlock&)            = delete;
    Bf16TmaDescriptorBlock& operator=(const Bf16TmaDescriptorBlock&) = delete;

    ~Bf16TmaDescriptorBlock() {
        if (device == nullptr) { return; }
        CUDA_CHECK(cudaFreeAsync(device, nullptr));
    }
};
#endif

template <class Schedule, class Output, class Epilogue>
void launch_bf16_a16_tma_mma(const Bf16A16Operands& p, Output output, Epilogue epilogue,
                             cudaStream_t stream) {
    // Descriptors are launch-owned values, copied into kernel parameters during Graph capture.
    const auto descriptors = make_bf16_tma_descriptors<Schedule>(p);
#ifdef _WIN32
    Bf16TmaDescriptorBlock block(stream);
    CUDA_CHECK(cudaMemcpyAsync(block.device, &descriptors, sizeof(descriptors),
                               cudaMemcpyHostToDevice, stream));
#endif
    for_each_token_slice(p.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const auto blocks = static_cast<std::int64_t>(p.rows / Schedule::kBlockRows) *
                            div_up(count, Schedule::kBlockTokens);
        if (blocks > 2147483647LL)
            throw std::invalid_argument("BF16 TMA grid exceeds CUDA grid.x capacity");
        const auto launch = [&]<bool Full>() {
            constexpr auto kernel = bf16_a16_tma_mma_kernel<Schedule, Full, Output, Epilogue>;
            constexpr int bytes =
                bf16_tma_scratch_bytes<Schedule, Epilogue> + Schedule::kBarrierBytes;
            bf16_prepare_shared<bytes, kernel>();
            kernel<<<static_cast<unsigned>(blocks), Schedule::kThreads, bytes, stream>>>(
#ifdef _WIN32
                block.device, output, epilogue, p.rows, p.k, offset, count);
#else
                descriptors, output, epilogue, p.rows, p.k, offset, count);
#endif
            CUDA_CHECK(cudaGetLastError());
        };
        if (count % Schedule::kBlockTokens == 0)
            launch.template operator()<true>();
        else
            launch.template operator()<false>();
    });
}

} // namespace ninfer::ops::detail
