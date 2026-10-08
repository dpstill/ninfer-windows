#pragma once

// ninfer::ops::detail - private launch prototype for log_softmax.

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void log_softmax_launch(const Tensor& logits, std::int32_t valid_rows, Tensor& output,
                        cudaStream_t stream);

} // namespace ninfer::ops::detail
