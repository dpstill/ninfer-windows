#include "corpus.h"
#include "evaluation.h"

#include "ninfer/engine.h"
#include "product/logging/logging.h"
#include "product/logging/pretty_format.h"
#include "product/logging/startup_log.h"

#include <nlohmann/json.hpp>
#include <spdlog/logger.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using json  = nlohmann::json;
using ninfer::perplexity::CorpusSelection;
using ninfer::perplexity::ScoreAggregate;
using ninfer::perplexity::WindowPlan;

struct Options {
    bool help_requested = false;
    std::filesystem::path artifact;
    std::optional<std::filesystem::path> corpus;
    std::optional<std::filesystem::path> text;
    std::optional<std::filesystem::path> output;
    std::uint32_t context               = 4096;
    std::uint32_t stride                = 2048;
    int device                          = 0;
    ninfer::KvCacheStorage kv           = ninfer::KvCacheStorage::Fp8E4M3Row256;
    bool quick                          = false;
    std::optional<std::uint32_t> gdn_nvfp4_layer;
    std::optional<std::filesystem::path> distributions;
    std::uint32_t distribution_positions = 8192;
    ninfer::product::LogLevel log_level = ninfer::product::LogLevel::Info;
};

std::string usage_text() {
    return "usage: ninfer-perplexity <model.ninfer> "
           "(--corpus <manifest.json> [--quick] | --text <utf8-file>)\n"
           "       [--context N] [--stride N] [--device N]\n"
           "       [--kv-dtype bf16|int8|fp8|nvfp4|k8v4] [--output <directory>]\n"
           "       [--gdn-nvfp4-layer N] [--distributions <directory> "
           "[--distribution-positions N]]\n"
           "       [--log-level trace|debug|info|warning|error|critical|off]\n";
}

[[noreturn]] void usage_error(std::string_view message) {
    throw std::invalid_argument(std::string(message));
}

template <class Integer>
Integer parse_integer(std::string_view text, const char* label) {
    Integer value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        usage_error(std::string("invalid ") + label + ": " + std::string(text));
    }
    return value;
}

Options parse_options(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        return Options{.help_requested = true};
    }
    if (argc < 2 || std::string_view(argv[1]).starts_with("--")) {
        usage_error("artifact path is required");
    }
    Options out;
    out.artifact = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string_view option = argv[i];
        const auto value              = [&](const char* label) -> std::string_view {
            if (++i >= argc) { usage_error(std::string(label) + " requires a value"); }
            return argv[i];
        };
        if (option == "--corpus") {
            out.corpus = std::filesystem::path(value("--corpus"));
        } else if (option == "--text") {
            out.text = std::filesystem::path(value("--text"));
        } else if (option == "--quick") {
            out.quick = true;
        } else if (option == "--context") {
            out.context = parse_integer<std::uint32_t>(value("--context"), "context");
        } else if (option == "--stride") {
            out.stride = parse_integer<std::uint32_t>(value("--stride"), "stride");
        } else if (option == "--device") {
            out.device = parse_integer<int>(value("--device"), "device");
        } else if (option == "--kv-dtype") {
            const std::string_view dtype = value("--kv-dtype");
            if (dtype == "bf16") {
                out.kv = ninfer::KvCacheStorage::BFloat16;
            } else if (dtype == "int8") {
                out.kv = ninfer::KvCacheStorage::Int8Group64;
            } else if (dtype == "fp8") {
                out.kv = ninfer::KvCacheStorage::Fp8E4M3Row256;
            } else if (dtype == "nvfp4") {
                out.kv = ninfer::KvCacheStorage::Nvfp4Group16;
            } else if (dtype == "k8v4") {
                out.kv = ninfer::KvCacheStorage::Fp8KeyNvfp4Value;
            } else {
                usage_error("--kv-dtype must be bf16, int8, fp8, nvfp4, or k8v4");
            }
        } else if (option == "--output") {
            out.output = std::filesystem::path(value("--output"));
        } else if (option == "--gdn-nvfp4-layer") {
            out.gdn_nvfp4_layer = parse_integer<std::uint32_t>(value("--gdn-nvfp4-layer"),
                                                               "gdn-nvfp4-layer");
        } else if (option == "--distributions") {
            out.distributions = std::filesystem::path(value("--distributions"));
        } else if (option == "--distribution-positions") {
            out.distribution_positions =
                parse_integer<std::uint32_t>(value("--distribution-positions"),
                                             "distribution-positions");
        } else if (option == "--log-level") {
            out.log_level = ninfer::product::parse_log_level(value("--log-level"));
        } else {
            usage_error("unknown option: " + std::string(option));
        }
    }
    if (out.corpus.has_value() == out.text.has_value()) {
        usage_error("exactly one of --corpus and --text is required");
    }
    if (out.quick && !out.corpus) { usage_error("--quick requires --corpus"); }
    if (out.context < 2 || out.stride == 0 || out.stride >= out.context) {
        usage_error("context/stride must satisfy context>=2 and 1<=stride<context");
    }
    if (out.distribution_positions == 0) {
        usage_error("--distribution-positions must be at least 1");
    }
    return out;
}

std::string kv_name(ninfer::KvCacheStorage value) {
    switch (value) {
    case ninfer::KvCacheStorage::BFloat16:
        return "bf16";
    case ninfer::KvCacheStorage::Int8Group64:
        return "int8-g64";
    case ninfer::KvCacheStorage::Fp8E4M3Row256:
        return "fp8-e4m3-r256";
    case ninfer::KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case ninfer::KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    }
    throw std::logic_error("unknown KV dtype");
}

std::string safe_component(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const unsigned char c : value) {
        out.push_back(std::isalnum(c) || c == '-' || c == '_' || c == '.' ? static_cast<char>(c)
                                                                          : '-');
    }
    return out.empty() ? "unknown" : out;
}

std::string timestamp() {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    std::ostringstream out;
    out << std::put_time(&utc, "%Y%m%d-%H%M%S");
    return out.str();
}

std::filesystem::path prepare_output_directory(const Options& options,
                                               const ninfer::LoadSummary& load,
                                               const CorpusSelection& corpus) {
    std::filesystem::path output = options.output.value_or(
        std::filesystem::path("profiles/perplexity") / safe_component(load.model_name) /
        safe_component(load.prefill_signature) / kv_name(options.kv) /
        safe_component(corpus.corpus_id) / safe_component(corpus.mode) / timestamp());
    if (std::filesystem::exists(output)) {
        if (!std::filesystem::is_directory(output) ||
            std::filesystem::directory_iterator(output) != std::filesystem::directory_iterator()) {
            throw std::runtime_error("output directory exists and is not empty: " +
                                     output.string());
        }
    } else if (!std::filesystem::create_directories(output)) {
        throw std::runtime_error("cannot create output directory: " + output.string());
    }
    return std::filesystem::absolute(output).lexically_normal();
}

double seconds_since(Clock::time_point begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
}

json aggregate_json(const ScoreAggregate& value) {
    return json{{"scored_tokens", value.scored_tokens},
                {"total_nll", value.total_nll},
                {"mean_nll", value.mean_nll()},
                {"perplexity", value.ppl()}};
}

struct EvaluationStream {
    ninfer::perplexity::CorpusStream source;
    std::vector<ninfer::TokenId> tokens;
    std::vector<WindowPlan> windows;
};

int run(const Options& options, const std::shared_ptr<spdlog::logger>& logger,
        ninfer::product::StartupLogRenderer& startup_log,
        const std::shared_ptr<ninfer::product::TerminalProgress>& progress) {
    const Clock::time_point total_started = Clock::now();
    ninfer::EngineOptions engine_options;
    engine_options.artifact_path    = options.artifact;
    engine_options.purpose          = ninfer::EnginePurpose::CausalScoring;
    engine_options.device           = options.device;
    engine_options.max_context      = options.context;
    engine_options.kv_cache         = options.kv;
    engine_options.gdn_nvfp4_layer  = options.gdn_nvfp4_layer;
    engine_options.startup_observer = startup_log.observer();
    ninfer::Engine engine(std::move(engine_options));
    const ninfer::LoadSummary load = engine.load_summary();
    startup_log.engine_ready(load);

    const Clock::time_point preflight_started = Clock::now();
    logger->info("preparing corpus");
    CorpusSelection corpus = options.corpus
                                 ? ninfer::perplexity::load_corpus(*options.corpus, options.quick)
                                 : ninfer::perplexity::load_custom_text(*options.text);
    std::vector<EvaluationStream> streams;
    streams.reserve(corpus.streams.size());
    std::uint64_t total_scored_tokens = 0;
    std::uint64_t total_input_tokens  = 0;
    std::uint64_t total_windows       = 0;
    for (auto& source : corpus.streams) {
        std::vector<ninfer::TokenId> tokens = engine.tokenize_text(source.text);
        if (tokens.size() < 2) {
            throw std::runtime_error("stream tokenized to fewer than two tokens: " + source.id);
        }
        std::vector<WindowPlan> windows =
            ninfer::perplexity::plan_windows(tokens.size(), options.context, options.stride);
        total_input_tokens += static_cast<std::uint64_t>(tokens.size());
        total_scored_tokens += static_cast<std::uint64_t>(tokens.size() - 1);
        total_windows += static_cast<std::uint64_t>(windows.size());
        streams.push_back(EvaluationStream{.source  = std::move(source),
                                           .tokens  = std::move(tokens),
                                           .windows = std::move(windows)});
    }
    const double preflight_seconds = seconds_since(preflight_started);
    logger->info("corpus ready | {} streams | {} input tokens | {} scored tokens | {} windows | {}",
                 ninfer::product::format_pretty_count(streams.size()),
                 ninfer::product::format_pretty_count(total_input_tokens),
                 ninfer::product::format_pretty_count(total_scored_tokens),
                 ninfer::product::format_pretty_count(total_windows),
                 ninfer::product::format_pretty_duration(preflight_seconds));

    const std::filesystem::path output_directory = prepare_output_directory(options, load, corpus);
    const Clock::time_point scoring_started      = Clock::now();
    logger->info("scoring | {} streams | {} tokens | {} windows",
                 ninfer::product::format_pretty_count(streams.size()),
                 ninfer::product::format_pretty_count(total_scored_tokens),
                 ninfer::product::format_pretty_count(total_windows));
    Clock::time_point next_progress = scoring_started + std::chrono::seconds(10);
    ScoreAggregate overall;
    std::map<std::string, ScoreAggregate> domains;
    json stream_reports             = json::array();
    std::uint64_t completed_windows = 0;

    const bool dump_distributions = options.distributions.has_value();
    std::filesystem::path distributions_directory;
    std::uint64_t distribution_budget = options.distribution_positions;
    std::uint64_t distribution_dumped = 0;
    std::uint32_t distribution_vocab  = 0;
    std::vector<std::ofstream> dist_bin, dist_targets, dist_positions;
    std::vector<std::uint64_t> dist_rows;
    if (dump_distributions) {
        distributions_directory =
            std::filesystem::absolute(*options.distributions).lexically_normal();
        if (!std::filesystem::exists(distributions_directory)) {
            std::filesystem::create_directories(distributions_directory);
        }
        dist_bin.resize(streams.size());
        dist_targets.resize(streams.size());
        dist_positions.resize(streams.size());
        dist_rows.resize(streams.size());
    }

    for (std::size_t stream_index = 0; stream_index < streams.size(); ++stream_index) {
        EvaluationStream& stream = streams[stream_index];
        std::ostringstream stream_status;
        stream_status << "  scoring [" << stream_index + 1 << '/' << streams.size() << "] "
                      << ninfer::product::format_pretty_text(stream.source.id) << " | "
                      << ninfer::product::format_pretty_count(stream.tokens.size()) << " tokens | "
                      << ninfer::product::format_pretty_count(stream.windows.size()) << " windows";
        if (progress->enabled()) {
            progress->update(stream_status.str());
        } else {
            logger->debug("{}", stream_status.str());
        }
        const Clock::time_point stream_started = Clock::now();
        ScoreAggregate stream_score;
        json window_reports = json::array();
        for (std::size_t window_index = 0; window_index < stream.windows.size(); ++window_index) {
            const WindowPlan& window = stream.windows[window_index];
            std::vector<ninfer::TokenId> input(
                stream.tokens.begin() + static_cast<std::ptrdiff_t>(window.input_begin),
                stream.tokens.begin() + static_cast<std::ptrdiff_t>(window.input_end));
            const std::size_t expected = window.target_end - window.target_begin;
            const Clock::time_point window_started = Clock::now();
            std::vector<float> logprobs;
            if (dump_distributions && distribution_budget > 0) {
                const std::size_t take = std::min<std::size_t>(distribution_budget, expected);
                ninfer::DistributionScore dist;
                try {
                    dist = engine.score_distributions(input, window.first_target);
                } catch (const std::exception& error) {
                    throw std::runtime_error("scoring " + stream.source.id + " window " +
                                             std::to_string(window_index) +
                                             " distributions failed: " + error.what());
                }
                if (dist.positions != expected ||
                    dist.logprobs.size() != static_cast<std::size_t>(dist.positions) *
                                                dist.vocab_size) {
                    throw std::runtime_error(
                        "scoring returned an invalid distribution shape for " + stream.source.id);
                }
                logprobs.resize(expected);
                for (std::size_t k = 0; k < expected; ++k) {
                    const std::uint32_t target_token = stream.tokens[window.target_begin + k];
                    logprobs[k] = dist.logprobs[k * dist.vocab_size + target_token];
                }
                if (take > 0 && !dist_bin[stream_index].is_open()) {
                    const std::string stem = "s" + std::to_string(stream_index);
                    dist_bin[stream_index].open(
                        distributions_directory / (stem + ".dist.bin"),
                        std::ios::binary | std::ios::app);
                    dist_targets[stream_index].open(
                        distributions_directory / (stem + ".targets.i32"),
                        std::ios::binary | std::ios::app);
                    dist_positions[stream_index].open(
                        distributions_directory / (stem + ".positions.i64"),
                        std::ios::binary | std::ios::app);
                    if (dist_bin[stream_index].fail() || dist_targets[stream_index].fail() ||
                        dist_positions[stream_index].fail()) {
                        throw std::runtime_error(
                            "cannot open distributions files in " + distributions_directory.string());
                    }
                }
                for (std::size_t k = 0; k < take; ++k) {
                    const std::uint32_t target_token = stream.tokens[window.target_begin + k];
                    const std::int64_t global_position =
                        static_cast<std::int64_t>(window.target_begin + k);
                    dist_bin[stream_index].write(
                        reinterpret_cast<const char*>(dist.logprobs.data() +
                                                       k * static_cast<std::size_t>(dist.vocab_size)),
                        static_cast<std::streamsize>(dist.vocab_size) *
                            static_cast<std::streamsize>(sizeof(float)));
                    dist_targets[stream_index].write(reinterpret_cast<const char*>(&target_token),
                                                     sizeof(target_token));
                    dist_positions[stream_index].write(
                        reinterpret_cast<const char*>(&global_position), sizeof(global_position));
                }
                dist_rows[stream_index] += take;
                distribution_budget -= take;
                distribution_dumped += take;
                distribution_vocab = dist.vocab_size;
            } else {
                try {
                    logprobs = engine.score_tokens(std::move(input), window.first_target);
                } catch (const std::exception& error) {
                    throw std::runtime_error("scoring " + stream.source.id + " window " +
                                             std::to_string(window_index) + " failed: " +
                                             error.what());
                }
            }
            if (logprobs.size() != expected) {
                throw std::runtime_error("scoring returned an invalid target count for " +
                                         stream.source.id);
            }
            ScoreAggregate window_score;
            window_score.add(logprobs);
            stream_score.add(window_score);
            overall.add(window_score);
            domains[stream.source.domain].add(window_score);
            ++completed_windows;
            json window_report            = aggregate_json(window_score);
            window_report["index"]        = window_index;
            window_report["input_begin"]  = window.input_begin;
            window_report["input_end"]    = window.input_end;
            window_report["target_begin"] = window.target_begin;
            window_report["target_end"]   = window.target_end;
            window_report["first_target"] = window.first_target;
            window_report["seconds"]      = seconds_since(window_started);
            window_reports.push_back(std::move(window_report));

            if (Clock::now() >= next_progress) {
                const double elapsed = seconds_since(scoring_started);
                const double rate    = static_cast<double>(overall.scored_tokens) / elapsed;
                const std::uint64_t remaining = total_scored_tokens - overall.scored_tokens;
                const double eta = rate > 0 ? static_cast<double>(remaining) / rate : 0.0;
                std::ostringstream line;
                line << "scoring | " << ninfer::product::format_pretty_count(overall.scored_tokens)
                     << '/' << ninfer::product::format_pretty_count(total_scored_tokens)
                     << " tokens | " << completed_windows << '/' << total_windows
                     << " windows | PPL " << std::fixed << std::setprecision(4) << overall.ppl()
                     << " | " << ninfer::product::format_pretty_rate(rate, "tok") << " | elapsed "
                     << ninfer::product::format_pretty_duration(elapsed) << " | ETA "
                     << ninfer::product::format_pretty_duration(eta);
                if (progress->enabled()) {
                    progress->update("  " + line.str());
                } else {
                    logger->info("{}", line.str());
                }
                next_progress = Clock::now() + std::chrono::seconds(10);
            }
        }
        const double stream_seconds = seconds_since(stream_started);
        progress->clear();
        logger->info("[{}/{}] {} | {} scored tokens | PPL {:.6g} | {}", stream_index + 1,
                     streams.size(), ninfer::product::format_pretty_text(stream.source.id),
                     ninfer::product::format_pretty_count(stream_score.scored_tokens),
                     stream_score.ppl(), ninfer::product::format_pretty_duration(stream_seconds));
        json stream_report               = aggregate_json(stream_score);
        stream_report["id"]              = stream.source.id;
        stream_report["domain"]          = stream.source.domain;
        stream_report["path"]            = stream.source.path.string();
        stream_report["input_tokens"]    = stream.tokens.size();
        stream_report["unscored_tokens"] = 1;
        stream_report["seconds"]         = stream_seconds;
        stream_report["windows"]         = std::move(window_reports);
        stream_reports.push_back(std::move(stream_report));
    }

    if (dump_distributions) {
        json dist_streams = json::array();
        for (std::size_t si = 0; si < streams.size(); ++si) {
            if (dist_rows[si] == 0) { continue; }
            const std::string stem = "s" + std::to_string(si);
            dist_streams.push_back(json{{"index", si},
                                        {"id", streams[si].source.id},
                                        {"rows", dist_rows[si]},
                                        {"bin", stem + ".dist.bin"},
                                        {"targets", stem + ".targets.i32"},
                                        {"positions", stem + ".positions.i64"}});
        }
        json dist_manifest{
            {"vocab_size", distribution_vocab},
            {"layout",
             "bin is row-major [rows][vocab_size] FP32; row p predicts the token at global "
             "position positions[p] (targets[p] is that token id)"},
            {"position_cap", options.distribution_positions},
            {"positions_dumped", distribution_dumped},
            {"streams", std::move(dist_streams)}};
        const std::filesystem::path manifest_path = distributions_directory / "manifest.json";
        std::ofstream manifest_out(manifest_path, std::ios::binary | std::ios::trunc);
        if (!manifest_out) {
            throw std::runtime_error("cannot create distributions manifest: " +
                                     manifest_path.string());
        }
        manifest_out << std::setw(2) << dist_manifest << '\n';
        manifest_out.flush();
        if (!manifest_out) {
            throw std::runtime_error("cannot write distributions manifest: " +
                                     manifest_path.string());
        }
        logger->info("distributions | {} positions | {}",
                     ninfer::product::format_pretty_count(distribution_dumped),
                     distributions_directory.string());
    }

    const double scoring_seconds = seconds_since(scoring_started);
    progress->clear();
    logger->info("scoring complete | {} tokens | {} windows | PPL {:.6g} | {} | {}",
                 ninfer::product::format_pretty_count(overall.scored_tokens), completed_windows,
                 overall.ppl(), ninfer::product::format_pretty_duration(scoring_seconds),
                 ninfer::product::format_pretty_rate(
                     static_cast<double>(overall.scored_tokens) / scoring_seconds, "tok"));
    json domain_reports = json::array();
    for (const auto& [domain, aggregate] : domains) {
        json item      = aggregate_json(aggregate);
        item["domain"] = domain;
        domain_reports.push_back(std::move(item));
    }

    json report{
        {"schema_version", 2},
        {"metric",
         {{"name", "fixed-window truncated-context causal perplexity"}, {"log_base", "natural"}}},
        {"artifact",
         {{"path", std::filesystem::absolute(options.artifact).lexically_normal().string()},
          {"architecture", load.architecture},
          {"name", load.model_name},
          {"prefill_signature", load.prefill_signature},
          {"formats", load.weight_formats}}},
        {"corpus",
         {{"id", corpus.corpus_id},
          {"mode", corpus.mode},
          {"source", corpus.source.string()},
          {"stream_count", streams.size()}}},
        {"execution",
          {{"purpose", "causal_scoring"},
           {"device", options.device},
           {"context_tokens", options.context},
           {"stride_tokens", options.stride},
           {"prefill_chunk_tokens", 1024},
           {"score_tile_tokens", 1024},
           {"kv_dtype", kv_name(options.kv)},
           {"gdn_nvfp4_layer",
            options.gdn_nvfp4_layer.has_value() ? json(*options.gdn_nvfp4_layer) : json()},
           {"distributions",
            options.distributions.has_value()
                ? json(
                      std::filesystem::absolute(*options.distributions).lexically_normal().string())
                : json()}}},
        {"timing",
         {{"load_seconds", load.load_seconds},
          {"read_and_tokenize_seconds", preflight_seconds},
          {"score_seconds", scoring_seconds},
          {"total_seconds", seconds_since(total_started)},
          {"scored_tokens_per_second",
           static_cast<double>(overall.scored_tokens) / scoring_seconds}}},
        {"streams", std::move(stream_reports)},
        {"domains", std::move(domain_reports)},
        {"overall", aggregate_json(overall)},
    };

    const std::filesystem::path temporary = output_directory / "report.json.tmp";
    const std::filesystem::path final     = output_directory / "report.json";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) { throw std::runtime_error("cannot create report: " + temporary.string()); }
        output << std::setw(2) << report << '\n';
        output.flush();
        if (!output) { throw std::runtime_error("cannot write report: " + temporary.string()); }
    }
    std::filesystem::rename(temporary, final);

    std::cout << "Perplexity result\n"
              << "artifact: " << load.model_name << '\n'
              << "kv: " << kv_name(options.kv) << ", corpus: " << corpus.corpus_id << " / "
              << corpus.mode << ", context/stride: " << options.context << '/' << options.stride
              << "\n\n";
    std::cout << std::left << std::setw(24) << "domain" << std::right << std::setw(16) << "tokens"
              << std::setw(16) << "mean_nll" << std::setw(16) << "ppl" << '\n';
    for (const auto& [domain, aggregate] : domains) {
        std::cout << std::left << std::setw(24) << domain << std::right << std::setw(16)
                  << aggregate.scored_tokens << std::setw(16) << std::fixed << std::setprecision(6)
                  << aggregate.mean_nll() << std::setw(16) << aggregate.ppl() << '\n';
    }
    std::cout << std::left << std::setw(24) << "overall" << std::right << std::setw(16)
              << overall.scored_tokens << std::setw(16) << std::fixed << std::setprecision(6)
              << overall.mean_nll() << std::setw(16) << overall.ppl() << "\n\n"
              << "score rate: " << std::setprecision(1)
              << static_cast<double>(overall.scored_tokens) / scoring_seconds << " tok/s\n"
              << "report: " << final << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    try {
        options = parse_options(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "ninfer-perplexity: " << error.what() << '\n';
        std::cerr << usage_text();
        return 1;
    }
    if (options.help_requested) {
        std::cout << usage_text();
        return 0;
    }

    ninfer::product::LoggingRuntime logging(
        {.logger_name  = "ninfer-perplexity",
         .level        = options.log_level,
         .presentation = ninfer::product::LogPresentation::Tool});
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    ninfer::product::StartupLogRenderer startup_log(logging);
    try {
        return run(options, logger, startup_log, logging.terminal_progress());
    } catch (const std::exception& error) {
        logging.terminal_progress()->clear();
        logger->error("{}", ninfer::product::format_pretty_text(error.what()));
        return 1;
    }
}
