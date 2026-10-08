#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Op: Full-vocabulary log-softmax
 *
 * Math / indexing:
 *   Let l[r,c] be the exact real value represented by logits[r,c]. For every column c and row
 *   r in [0,valid_rows),
 *
 *     ideal[r,c] = l[r,c] - log(sum_{r'=0..valid_rows-1} exp(l[r',c])).
 *
 * Logical shapes:
 *   logits is [physical_rows,C], output is [valid_rows,C], with C>0 and 1<=valid_rows<=
 *   physical_rows. Physical rows [valid_rows,physical_rows) do not participate in either the
 *   denominator or the output.
 *
 * Supported domain:
 *   logits is contiguous finite BF16, and output is contiguous FP32. Storage has its dtype's
 *   natural alignment.
 *
 * Numeric:
 *   output is the FP32 numerical approximation of ideal, computed through the stable
 *   subtract-column-maximum form. Reduction association and private accumulator precision are
 *   implementation choices; the independent oracle evaluates the full formula in FP64 from the
 *   represented BF16 inputs.
 *
 * Effects:
 *   Writes every output element and preserves logits. Output must not overlap logits.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Enqueues work on stream and owns no persistent state.
 */
void log_softmax(const Tensor& logits, std::int32_t valid_rows, Tensor& output,
                 cudaStream_t stream);

} // namespace ninfer::ops
