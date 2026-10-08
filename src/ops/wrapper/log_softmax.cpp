// ninfer::ops - log_softmax wrapper: public contract validation and launcher dispatch.
#include "ninfer/ops/log_softmax.h"

#include "ops/launcher/log_softmax.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_rank_two(const Tensor& tensor, const char* label) {
    if (tensor.ne[0] <= 0 || tensor.ne[1] <= 0 || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("log_softmax: ") + label +
                                    " must be rank-2 with positive dimensions");
    }
}

void require_accessible(const Tensor& tensor, std::size_t alignment, const char* label) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("log_softmax: ") + label +
                                    " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string("log_softmax: ") + label +
                                    " data must be non-null");
    }
    if ((reinterpret_cast<std::uintptr_t>(tensor.data) & (alignment - 1)) != 0) {
        throw std::invalid_argument(std::string("log_softmax: ") + label +
                                    " data is not naturally aligned");
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    if (lhs_begin <= rhs_begin) { return rhs_begin - lhs_begin < lhs.bytes(); }
    return lhs_begin - rhs_begin < rhs.bytes();
}

} // namespace

void log_softmax(const Tensor& logits, std::int32_t valid_rows, Tensor& output,
                 cudaStream_t stream) {
    if (logits.dtype != DType::BF16) {
        throw std::invalid_argument("log_softmax: logits must be BF16");
    }
    if (output.dtype != DType::FP32) {
        throw std::invalid_argument("log_softmax: output must be FP32");
    }

    require_rank_two(logits, "logits");
    require_rank_two(output, "output");
    if (logits.ne[1] != output.ne[1]) {
        throw std::invalid_argument("log_softmax: output columns must match logits columns");
    }
    if (valid_rows <= 0 || valid_rows > logits.ne[0] || valid_rows != output.ne[0]) {
        throw std::invalid_argument("log_softmax: valid_rows must equal output rows and be in "
                                    "[1,physical_rows]");
    }

    (void)logits.bytes();
    (void)output.bytes();
    require_accessible(logits, alignof(std::uint16_t), "logits");
    require_accessible(output, alignof(float), "output");
    if (overlaps(output, logits)) {
        throw std::invalid_argument("log_softmax: output must not overlap logits");
    }

    detail::log_softmax_launch(logits, valid_rows, output, stream);
}

} // namespace ninfer::ops
