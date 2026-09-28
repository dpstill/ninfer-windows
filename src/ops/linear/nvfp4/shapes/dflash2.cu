// DFlash2 drafter matrices for the NVFP4-encoded draft module: weight-only A16 routes
// (gemv single token, exact small-T families, 32-token chunks for wider extents). The draft
// never quantizes activations, so no A4 route is registered for these shapes.
#include "ops/linear/nvfp4/nvfp4_shapes.h"
#include "ops/linear/nvfp4/nvfp4_dflash2_geometry.h"
#include "ops/linear/nvfp4/nvfp4_launch.cuh"

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Gemv =
    Nvfp4A16GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
template <int Tokens>
using Exact = Nvfp4A16SimtSchedule<(Tokens >= 8 && Tokens <= 16) ? 16 : 4, 1, 2, 16, Tokens, 1,
                                   Nvfp4SimtActivationAccess::TokenPacked, Nvfp4ScaleAccess::Direct,
                                   Nvfp4CodeCache::Default, 1, Nvfp4SimtBlockOrder::RowsContiguous,
                                   1>;
using C2  = Nvfp4A16SimtSchedule<4, 1, 2, 16, 2, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                 Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                 Nvfp4SimtBlockOrder::RowsContiguous, 1>;
using C4  = Nvfp4A16SimtSchedule<4, 1, 2, 16, 4, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                 Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                 Nvfp4SimtBlockOrder::RowsContiguous, 1>;
using C32 = Nvfp4A16SimtSchedule<4, 1, 2, 16, 16, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                 Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                 Nvfp4SimtBlockOrder::TokenTilesContiguous, 3>;
using FullChunk = Nvfp4A16SimtSchedule<4, 1, 2, 16, 32, 1, Nvfp4SimtActivationAccess::TokenPacked,
                                       Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                       Nvfp4SimtBlockOrder::RowsContiguous, 1>;

// The unified A16 launchers dropped the shape-level exact/chunk helpers the DFlash2 shapes were
// written against; keep local copies built on the A16 wrappers.
template <class Geometry, int First, int Last, template <int> class Schedule, std::size_t... I>
constexpr auto dflash2_exact_launchers(std::index_sequence<I...>) {
    return std::array<Nvfp4Launch, sizeof...(I)>{
        &nvfp4_linear_a16_simt<Geometry, static_cast<int>(First) + static_cast<int>(I),
                               Schedule<First + static_cast<int>(I)>, true>...};
}

template <class Geometry, int First, int Last, template <int> class Schedule>
Nvfp4Launch select_dflash2_exact(std::int32_t tokens) {
    static constexpr auto launchers =
        dflash2_exact_launchers<Geometry, First, Last, Schedule>(
            std::make_index_sequence<Last - First + 1>{});
    return launchers.at(static_cast<std::size_t>(tokens - First));
}

template <int Chunk, Nvfp4Launch (*Select)(std::int32_t)>
void launch_dflash2_a16_chunks(const Tensor& x, const Weight& weight, Tensor& out,
                               cudaStream_t stream) {
    for (std::int32_t offset = 0; offset < x.ne[1]; offset += Chunk) {
        const std::int32_t count = std::min(Chunk, x.ne[1] - offset);
        auto input               = x.slice(1, offset, count);
        auto output              = out.slice(1, offset, count);
        Select(count)(input, weight, output, stream);
    }
}

template <class Geometry>
Nvfp4Launch select_a16(std::int32_t tokens) {
    if (tokens == 1) return nvfp4_linear_a16_gemv<Geometry, Gemv>;
    if (tokens == 32) return nvfp4_linear_a16_simt<Geometry, 32, FullChunk, true>;
    if (tokens >= 5 && tokens <= 28) return select_dflash2_exact<Geometry, 5, 28, Exact>(tokens);
    if (tokens <= 2) return nvfp4_linear_a16_simt<Geometry, 2, C2, true>;
    if (tokens <= 4) return nvfp4_linear_a16_simt<Geometry, 4, C4, false>;
    if (tokens <= 32) return nvfp4_linear_a16_simt<Geometry, 32, C32, false>;
    throw std::logic_error("nvfp4 DFlash2 A16 chunk exceeds shape capacity");
}

template <class Geometry>
const Nvfp4LinearShape make_dflash2_shape() {
    return {Geometry::kOutputRows, Geometry::kInputRows,
            launch_dflash2_a16_chunks<32, select_a16<Geometry>>, nullptr,
            [](std::int32_t, std::int32_t) { return false; }};
}

using Feature  = Nvfp4Geometry<5120, 25600>;
using Qkv      = Nvfp4Geometry<6144, 5120>;
using AttnOut  = Nvfp4Geometry<5120, 4096>;
using ConvProj = Nvfp4Geometry<1280, 5120>;
using Selector = Nvfp4Geometry<256, 5120>;

} // namespace

const Nvfp4LinearShape kNvfp4DFlash2Feature  = make_dflash2_shape<Feature>();
const Nvfp4LinearShape kNvfp4DFlash2Qkv      = make_dflash2_shape<Qkv>();
const Nvfp4LinearShape kNvfp4DFlash2AttnOut  = make_dflash2_shape<AttnOut>();
const Nvfp4LinearShape kNvfp4DFlash2ConvProj = make_dflash2_shape<ConvProj>();
const Nvfp4LinearShape kNvfp4DFlash2Selector = make_dflash2_shape<Selector>();

} // namespace ninfer::ops::detail
