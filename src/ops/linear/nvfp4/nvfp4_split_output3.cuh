#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Three-destination split for parents whose rows partition into query, key, and value segments.
// Row-group vectors never straddle a segment boundary when both row counts are multiples of eight.
template <std::int32_t QueryRows, std::int32_t KvRows>
struct Nvfp4SplitOutput3 {
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* value;
    static_assert((QueryRows % 8) == 0 && (KvRows % 8) == 0);

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float result) const {
        if (parent_row < QueryRows) {
            query[static_cast<std::int64_t>(token) * QueryRows + parent_row] =
                __float2bfloat16_rn(result);
        } else if (parent_row < QueryRows + KvRows) {
            key[static_cast<std::int64_t>(token) * KvRows + parent_row - QueryRows] =
                __float2bfloat16_rn(result);
        } else {
            value[static_cast<std::int64_t>(token) * KvRows + parent_row - QueryRows - KvRows] =
                __float2bfloat16_rn(result);
        }
    }
};

} // namespace ninfer::ops::detail
