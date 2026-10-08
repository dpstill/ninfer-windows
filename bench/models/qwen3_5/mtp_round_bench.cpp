#include "runtime/engine/model_instance.h"

#include "core/device.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/kv_capacity.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace qwen = ninfer::models::qwen3_5;

struct Options {
    std::filesystem::path artifact = "out/qwen3_6_27b.ninfer";
    int device                     = 0;
    int warmup                     = 2;
    int repetitions                = 10;
    std::uint32_t draft_tokens     = 5;
    ninfer::ProposalHead proposal  = ninfer::ProposalHead::Optimized;
    bool use_cuda_graph            = true;
    ninfer::KvCacheStorage kv_dtype = ninfer::KvCacheStorage::BFloat16;
    std::uint32_t context          = 6;
};

void print_usage(const char* executable) {
    std::cout << "usage: " << executable
              << " [--artifact <model.ninfer>] [--device <id>] [--warmup <n>] [--reps <n>]"
                  " [--draft-tokens <1..7>] [--proposal-head full|optimized]"
                 " [--kv-dtype bf16|nvfp4] [--context <n>] [--no-cuda-graph]\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto value = [&](const char* name) -> const char* {
            if (++index >= argc) {
                throw std::invalid_argument(std::string(name) + " needs value");
            }
            return argv[index];
        };
        if (argument == "--artifact") {
            options.artifact = value("--artifact");
        } else if (argument == "--device") {
            options.device = std::stoi(value("--device"));
        } else if (argument == "--warmup") {
            options.warmup = std::stoi(value("--warmup"));
        } else if (argument == "--reps") {
            options.repetitions = std::stoi(value("--reps"));
        } else if (argument == "--draft-tokens") {
            options.draft_tokens = static_cast<std::uint32_t>(std::stoul(value("--draft-tokens")));
        } else if (argument == "--proposal-head") {
            const std::string_view head(value("--proposal-head"));
            if (head == "full") {
                options.proposal = ninfer::ProposalHead::Full;
            } else if (head == "optimized") {
                options.proposal = ninfer::ProposalHead::Optimized;
            } else {
                throw std::invalid_argument("--proposal-head must be full or optimized");
            }
        } else if (argument == "--kv-dtype") {
            const std::string_view dtype(value("--kv-dtype"));
            if (dtype == "bf16") {
                options.kv_dtype = ninfer::KvCacheStorage::BFloat16;
            } else if (dtype == "nvfp4") {
                options.kv_dtype = ninfer::KvCacheStorage::Nvfp4Group16;
            } else {
                throw std::invalid_argument("--kv-dtype must be bf16 or nvfp4");
            }
        } else if (argument == "--context") {
            options.context = static_cast<std::uint32_t>(std::stoul(value("--context")));
        } else if (argument == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (argument == "-h" || argument == "--help") {
            print_usage(argc > 0 ? argv[0] : "ninfer_qwen3_5_mtp_round_bench");
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (options.device < 0) { throw std::invalid_argument("--device must be nonnegative"); }
    if (options.warmup < 0) { throw std::invalid_argument("--warmup must be nonnegative"); }
    if (options.repetitions <= 0) { throw std::invalid_argument("--reps must be positive"); }
    if (options.draft_tokens == 0 || options.draft_tokens > 7) {
        throw std::invalid_argument("--draft-tokens must be in [1,7]");
    }
    if (options.context == 0) {
        throw std::invalid_argument("--context must be positive");
    }
    return options;
}

struct RoundMeasurement {
    float milliseconds            = 0.0F;
    std::uint32_t licensed_tokens = 0;
    std::uint32_t frontier        = 0;
    ninfer::SpeculativeStats stats{};
};

RoundMeasurement measure_round(qwen::Program& program, ninfer::DeviceContext& device,
                               qwen::SequenceHandle sequence, std::uint32_t draft_tokens) {
    const std::array<qwen::SequenceHandle, 1> sequences{sequence};
    // A 2*(K+1) budget keeps the next round's MTP window at the full configured K: after a
    // fully-accepted round (K+1 licensed) at least K+1 tokens remain, so mtp_prepare_next_round
    // clamps to K instead of shrinking the window (which a K+1 budget would cause).
    const std::array<ninfer::runtime::RoundBudget, 1> budgets{
        ninfer::runtime::RoundBudget{.generated_tokens_remaining = 2U * (draft_tokens + 1U)}};
    ninfer::CudaEventTimer timer(device);
    timer.start();
    auto pending                 = program.decode(sequences, budgets);
    const std::uint32_t licensed = pending.row_counts().empty()
                                        ? 1U
                                        : static_cast<std::uint32_t>(pending.row_counts().front());
    const std::array<ninfer::runtime::CommitDecision, 1> decisions{
        ninfer::runtime::CommitDecision{.accepted_tokens = licensed}};
    const auto committed         = program.commit(std::move(pending), decisions);
    const float milliseconds      = timer.stop_ms();
    return RoundMeasurement{.milliseconds    = milliseconds,
                            .licensed_tokens = licensed,
                            .frontier        = 0,
                            .stats           = committed.rows[0].speculative};
}

// Creates a new sequence from the seed, runs prefill, and returns the active sequence handle.
// All state setup is outside the timed region.
qwen::SequenceHandle create_sequence(qwen::Program& program, qwen::Frontend& frontend,
                                     const std::vector<ninfer::TokenId>& seed,
                                     std::uint32_t draft_tokens, std::uint32_t measured_rounds) {
    auto prompt = frontend.prepare_tokens(seed, false);
    ninfer::runtime::ResolvedExecutionOptions execution;
    execution.requested_output_tokens = static_cast<std::uint32_t>(
        1ULL + static_cast<std::uint64_t>(measured_rounds) * (draft_tokens + 1ULL));
    execution.allow_prefix_reuse      = false;
    auto request_base = program.plan_request(prompt, execution);
    auto request_plan =
        program.inspect_admission(prompt, request_base, ninfer::runtime::LaneId{0}, nullptr,
                                  nullptr, std::nullopt, false);
    if (!request_plan) { throw std::runtime_error("sequence admission was rejected"); }
    auto resource_plan = program.seal_identity(*request_plan, prompt, {});
    if (!resource_plan) { throw std::runtime_error("sequence resources were not sealed"); }
    const auto reserved =
        program.start_resource_transaction(std::move(*resource_plan), std::move(prompt), {});
    if (reserved != ninfer::runtime::ContextTransactionReserveStatus::Reserved) {
        throw std::runtime_error("sequence materialization was not reserved");
    }
    std::optional<qwen::MaterializationResult> published;
    for (;;) {
        auto transaction = program.progress_context_transaction({});
        if (std::holds_alternative<ninfer::runtime::ContextTransactionInProgress>(transaction)) {
            continue;
        }
        if (!std::holds_alternative<qwen::MaterializationResult>(transaction)) {
            program.finalize_context_transaction();
            throw std::runtime_error("sequence returned the wrong transaction result");
        }
        published.emplace(std::get<qwen::MaterializationResult>(std::move(transaction)));
        break;
    }
    if (published->status != ninfer::runtime::ContextTransactionStatus::Published ||
        !published->published) {
        program.finalize_context_transaction();
        throw std::runtime_error("sequence materialization was not published");
    }
    auto started = std::move(*published->published);
    program.finalize_context_transaction();
    // Drive the production prefill scheduling loop until the entire seed context is ready.
    // Chunked prefill processes up to prefill_chunk tokens per call; a 2048-token context
    // with chunk=128 requires 16 scheduling units. Capture offers are ignored (optional
    // CUDA Graph optimization that does not block prefill completion).
    const std::uint32_t max_prefill_units = 1024;
    std::uint32_t prefill_units           = 0;
    for (;;) {
        auto progress = program.advance_prefill(started.sequence);
        if (progress.complete) {
            if (!progress.pending) {
                throw std::runtime_error("completed prefill has no pending batch");
            }
            const std::array<ninfer::runtime::CommitDecision, 1> begin_decision{
                ninfer::runtime::CommitDecision{.accepted_tokens = 1}};
            (void)program.commit(std::move(*progress.pending), begin_decision);
            break;
        }
        if (++prefill_units >= max_prefill_units) {
            throw std::runtime_error("prefill did not complete within 1024 scheduling units");
        }
    }
    return started.sequence;
}

int run(const Options& options) {
    if (!std::filesystem::exists(options.artifact)) {
        std::cout << "SKIP: artifact not present: " << options.artifact.string() << '\n';
        return 0;
    }

    // Build seed: first 6 tokens are the fixed pattern, remaining are filler token 248045.
    const std::vector<ninfer::TokenId> base_seed{248045, 846, 198, 5834, 248046, 198};
    std::vector<ninfer::TokenId> seed;
    seed.reserve(options.context);
    for (std::uint32_t i = 0; i < options.context; ++i) {
        seed.push_back(i < base_seed.size() ? base_seed[i] : 248045);
    }

    // A single sequence is prefilled once; warmup and measured rounds run consecutively on it.
    const std::uint32_t measured_rounds =
        static_cast<std::uint32_t>(options.warmup + options.repetitions);
    const std::uint64_t block     = options.draft_tokens + 1ULL;
    const std::uint64_t max_context =
        static_cast<std::uint64_t>(options.context) + measured_rounds * block + 64ULL;
    if (max_context > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("context and measured rounds exceed native capacity");
    }

    ninfer::EngineOptions engine;
    engine.artifact_path       = options.artifact;
    engine.device              = options.device;
    engine.max_context         = static_cast<std::uint32_t>(max_context);
    engine.kv_capacity         = ninfer::KvCapacityPolicy::explicit_capacity(engine.max_context);
    engine.prefill_chunk       = 128;
    engine.kv_cache            = options.kv_dtype;
    engine.speculative.backend = ninfer::SpeculativeBackend::Mtp;
    engine.speculative.draft_tokens  = options.draft_tokens;
    engine.speculative.proposal_head = options.proposal;
    engine.use_cuda_graph            = options.use_cuda_graph;

    engine = ninfer::runtime::normalize_engine_options(std::move(engine));
    ninfer::DeviceContext device(options.device);
    auto constructed = ninfer::runtime::construct_model(engine, device);
    auto& frontend   = constructed.instance->frontend;
    auto& program    = constructed.instance->program;

    // Prefill + Begin commit run once, outside the timed region.
    auto sequence = create_sequence(*program, frontend, seed, options.draft_tokens, measured_rounds);

    // Tracked execution frontier: the prompt context after the Begin commit, advanced by the
    // licensed tokens of every round.
    std::uint32_t frontier = options.context;

    RoundMeasurement warmup_state{};
    for (int i = 0; i < options.warmup; ++i) {
        warmup_state = measure_round(*program, device, sequence, options.draft_tokens);
        frontier += warmup_state.licensed_tokens;
    }

    std::vector<RoundMeasurement> measurements;
    measurements.reserve(static_cast<std::size_t>(options.repetitions));
    for (int i = 0; i < options.repetitions; ++i) {
        const std::uint32_t pre_round_frontier = frontier;
        auto result                            = measure_round(*program, device, sequence,
                                                                options.draft_tokens);
        frontier += result.licensed_tokens;
        result.frontier = pre_round_frontier;
        measurements.push_back(std::move(result));
    }

    // Sanity: the measured rounds must stay on the full-K, non-fallback, non-budget-limited path.
    const auto& before = warmup_state.stats;
    const auto& after  = measurements.back().stats;
    if (after.fallback_steps != before.fallback_steps) {
        throw std::runtime_error("MTP benchmark fallback_steps increased during measured rounds");
    }
    for (std::size_t i = 0; i < measurements.size(); ++i) {
        const auto previous_drafted =
            (i == 0) ? before.drafted_tokens : measurements[i - 1].stats.drafted_tokens;
        const auto proposed = measurements[i].stats.drafted_tokens - previous_drafted;
        if (proposed != options.draft_tokens) {
            throw std::runtime_error(
                "MTP benchmark proposed draft window deviated from the configured K");
        }
    }

    const auto aborted = program->abort(sequence);
    if (aborted.status != ninfer::runtime::ConsumeStatus::Consumed) {
        throw std::runtime_error("benchmark could not release sequence");
    }

    std::vector<float> milliseconds;
    milliseconds.reserve(measurements.size());
    std::uint64_t licensed_tokens = 0;
    std::uint32_t frontier_min    = measurements.front().frontier;
    std::uint32_t frontier_max    = measurements.front().frontier;
    for (const RoundMeasurement& measurement : measurements) {
        milliseconds.push_back(measurement.milliseconds);
        licensed_tokens += measurement.licensed_tokens;
        frontier_min     = std::min(frontier_min, measurement.frontier);
        frontier_max     = std::max(frontier_max, measurement.frontier);
    }
    const double mean_ms =
        std::accumulate(milliseconds.begin(), milliseconds.end(), 0.0) / measurements.size();
    const auto [minimum, maximum] = std::minmax_element(milliseconds.begin(), milliseconds.end());
    const double mean_licensed =
        static_cast<double>(licensed_tokens) / static_cast<double>(measurements.size());

    // Median and p95
    std::vector<float> sorted_ms(milliseconds);
    std::sort(sorted_ms.begin(), sorted_ms.end());
    const std::size_t n = sorted_ms.size();
    const float median_ms =
        (n % 2 == 1) ? sorted_ms[n / 2] : 0.5F * (sorted_ms[n / 2 - 1] + sorted_ms[n / 2]);
    const float p95_ms = sorted_ms[std::min<std::size_t>(n * 95 / 100, n - 1)];

    // Total licensed drift: tokens the context advanced across the measured rounds.
    const std::uint64_t licensed_drift = licensed_tokens;

    std::cout << "format,ninfer_qwen3_5_mtp_round_bench_v4\n";
    std::cout << "artifact," << options.artifact.string() << '\n';
    std::cout << "device," << device.props.name << '\n';
    std::cout << "draft_tokens," << options.draft_tokens << '\n';
    std::cout << "kv_dtype,"
               << (options.kv_dtype == ninfer::KvCacheStorage::Nvfp4Group16 ? "nvfp4" : "bf16")
               << '\n';
    std::cout << "context," << options.context << '\n';
    std::cout << "proposal_head,"
               << (options.proposal == ninfer::ProposalHead::Optimized ? "optimized" : "full")
               << '\n';
    std::cout << "cuda_graph," << (options.use_cuda_graph ? "true" : "false") << '\n';
    std::cout << "warmup," << options.warmup << '\n';
    std::cout << "repetitions," << options.repetitions << '\n';
    std::cout << "mtp_round_mean_ms," << mean_ms << '\n';
    std::cout << "mtp_round_median_ms," << median_ms << '\n';
    std::cout << "mtp_round_p95_ms," << p95_ms << '\n';
    std::cout << "mtp_round_min_ms," << *minimum << '\n';
    std::cout << "mtp_round_max_ms," << *maximum << '\n';
    std::cout << "mean_licensed_tokens," << mean_licensed << '\n';
    std::cout << "measured_frontier_min," << frontier_min << '\n';
    std::cout << "measured_frontier_max," << frontier_max << '\n';
    std::cout << "total_licensed_drift," << licensed_drift << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "ninfer_qwen3_5_mtp_round_bench: " << error.what() << '\n';
        return 1;
    }
}
