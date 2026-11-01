#include "engine/chat.hpp"
#include "engine/utils.hpp"
#include "engine/repetition_penalty.hpp"

#include "tensor/linalg.hpp"
#include "model/qgpt_bin_reader.hpp"
#include "model/riverstrike/riverstrike.hpp"
#include "util/chat_message.hpp"
#include "util/utf8.hpp"
#include "util/timestamp.hpp"
#include "util/fs.hpp"

#include <optional>
#include <unordered_set>

namespace quetzal::mode {

static void chat_mode_help() {
    std::cout << "[Quetzal] chat mode help:\n"
              << "     /exit    - exit chat mode\n"
              << "  /q /quit    - exit chat mode\n"
              << "  /c /clear   - clear chat history\n"
              << "  /h /help    - show help message\n";
}

static std::mt19937_64 create_generator(int seed) {
    if (seed == -1) {
        return std::mt19937_64(std::random_device{}());
    }
    return std::mt19937_64(seed);
}

void chat_mode(const quetzal::util::cli& cli) {
    util::log_manager logger(cli.enable_debug_info());

    gpt::qgpt_bin_reader qbr(cli.get_model_file_path(), cli.enable_debug_info());
    const auto& wm = qbr.get_weights_manager();
    const auto& br = qbr.get_bin_reader();
    bbpe::tokenizer tokenizer(br);
    repetition_penalty rp(cli.get_repetition_penalty(),
                          cli.get_frequency_penalty(),
                          cli.get_presence_penalty());

    const auto& cfg = qbr.get_model_config();
    auto max_seq_len = cfg.max_seq_len;

    std::optional<tensor::NTK_aware_scaling> rule;
    if (cli.get_rope_scaling_factor() > 1.f) {
        rule.emplace(tensor::NTK_aware_scaling(cli.get_rope_scaling_factor()));
        max_seq_len *= cli.get_rope_scaling_factor();
    }
    gpt::riverstrike_jishui_v1 model(wm, cfg, rule ? &*rule : nullptr);
    qbr.release_memory();
    std::mt19937_64 gen = create_generator(cli.get_random_seed());

    util::message_manager mm(tokenizer);
    if (!cli.system_prompt_disabled()) {
        mm.push("system",
            cli.get_system_prompt().empty()
                ? cfg.default_system_prompt
                : cli.get_system_prompt()
        );
    }

    utf8::utf8_stream_decoder decoder;

    info_dump(std::cout, cli, cfg, max_seq_len);

    const auto im_end = br.get_vocab_index().at("<|im_end|>");
    while (true) {
        std::string input;
        std::cout << ">>> " << std::flush;
        if (!std::getline(std::cin, input)) {
            break;
        }

        if (input == "/exit" || input == "/quit" || input == "/q") {
            break;
        } else if (input == "/c" || input == "/clear") {
            logger.log(mm.generate_log());
            mm.clear();
            continue;
        } else if (input == "/h" || input == "/help") {
            chat_mode_help();
            continue;
        }

        mm.push("user", input);

        // reserve 256 token for reply
        std::string prompt = mm.build(max_seq_len - 256, cli.enable_debug_info());
        std::vector<std::uint32_t> indices = tokenizer.encode(prompt);
        std::size_t index = indices.back();
        std::size_t prompt_end = indices.size();
        std::string output_content = "";
        bool prefill_executed = false;

        if (indices.size() >= max_seq_len) {
            std::cout << "[Info] warning: too long prompt: " << indices.size() << "\n\n";
        }

        util::timestamp ts;
        float TTFT = 0.f;
        ts.stamp();

        std::cout << "\n[Quetzal]\n";
        while (indices.size() < max_seq_len) {
            auto logits = prefill_executed ? model.decode(index) : model.prefill(indices);
            if (!prefill_executed) {
                TTFT = ts.elapsed_micro_seconds().count() / 1000000.f;
                ts.stamp();
            }
            prefill_executed = true;

            rp.apply(indices, logits, prompt_end);
            logits = logits / cli.get_temperature();
            tensor::apply_topk_mask(logits, cli.get_top_k());
            tensor::apply_topp_mask(logits, cli.get_top_p());
            auto topk = tensor::softmax<float>(logits);
            index = tensor::multinomial<float>(topk, gen);
            if (index == im_end) {
                break;
            }

            indices.push_back(index);
            auto token = decoder.feed(br.get_vocab()[index]);
            output_content += token;
            std::cout << token << std::flush;
        }
        if (indices.size() >= max_seq_len) {
            std::cout << "\n[Info] context out of range, truncated.\n";
        }

        mm.push("assistant", output_content);

        const auto dur = ts.elapsed_micro_seconds().count() / 1000000.f;
        const auto context_used = mm.get_token_count();
        std::cout << "\n[Info] (perf) TTFT: ";
        if (TTFT > 1.f) {
            std::cout << TTFT << "s, ";
        } else {
            std::cout << TTFT * 1000.f << "ms, ";
        }
        std::cout << "decode: ";
        std::cout << (indices.size() - prompt_end) * 1.f / dur << " token/s, ";
        std::cout << context_used * 100 / max_seq_len << "% used ("
                  << context_used << " tokens)\n";
    }
    if (!mm.empty()) {
        logger.log(mm.generate_log());
    }
}

}
