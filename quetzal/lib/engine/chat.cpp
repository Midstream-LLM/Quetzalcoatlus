#include "engine/chat.hpp"
#include "engine/utils.hpp"
#include "engine/runtime.hpp"

#include "tensor/linalg.hpp"
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
    runtime::runtime rt(cli);
    auto& model = rt.get_model();

    std::mt19937_64 gen = create_generator(cli.get_random_seed());

    util::message_manager mm(rt.get_tokenizer());
    if (!cli.system_prompt_disabled()) {
        mm.push("system",
            cli.get_system_prompt().empty()
                ? rt.get_model_config().default_system_prompt
                : cli.get_system_prompt()
        );
    }

    utf8::utf8_stream_decoder decoder;

    info_dump(std::cout, cli, rt.get_model_config(), rt.get_max_seq_len());

    const auto im_end = rt.get_vocab_index("<|im_end|>");
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
        std::string prompt = mm.build(rt.get_max_seq_len() - 256, cli.enable_debug_info());
        std::vector<std::uint32_t> indices = rt.get_tokenizer().encode(prompt);
        std::size_t index = indices.back();
        std::size_t prompt_end = indices.size();
        std::string output_content = "";
        bool prefill_executed = false;

        if (indices.size() >= rt.get_max_seq_len()) {
            std::cout << "[Info] warning: too long prompt: " << indices.size() << "\n\n";
        }

        util::timestamp ts;
        float TTFT = 0.f;
        ts.stamp();

        std::cout << "\n[Quetzal]\n";
        while (indices.size() < rt.get_max_seq_len()) {
            auto logits = prefill_executed ? model.decode(index) : model.prefill(indices);
            if (!prefill_executed) {
                TTFT = ts.elapsed_micro_seconds().count() / 1000000.f;
                ts.stamp();
            }
            prefill_executed = true;

            rt.get_penalty_applier().apply(indices, logits, prompt_end);
            logits = logits / cli.get_temperature();
            tensor::apply_topk_mask(logits, cli.get_top_k());
            tensor::apply_topp_mask(logits, cli.get_top_p());
            auto topk = tensor::softmax<float>(logits);
            index = tensor::multinomial<float>(topk, gen);
            if (index == im_end) {
                break;
            }

            indices.push_back(index);
            auto token = decoder.feed(rt.get_vocab(index));
            output_content += token;
            std::cout << token << std::flush;
        }
        if (indices.size() >= rt.get_max_seq_len()) {
            std::cout << "\n[Info] context out of range ("
                      << indices.size() << " >= " << rt.get_max_seq_len()
                      << "), truncated.\n";
        }

        mm.push("assistant", output_content);

        const auto dur_micro = ts.elapsed_micro_seconds().count();
        const auto dur = dur_micro / 1000000.f;
        const auto context_used = mm.get_token_count();
        std::cout << "\n[Info] (perf) TTFT: ";
        if (TTFT > 1.f) {
            std::cout << TTFT << "s, ";
        } else {
            std::cout << TTFT * 1000.f << "ms, ";
        }
        const auto TPOT = dur_micro * 1.f / (indices.size() - prompt_end);
        std::cout << "TPOT: ";
        if (TPOT > 1000000.f) {
            std::cout << TPOT / 1000000.f << "s/token";
        } else if (TPOT > 1000.f) {
            std::cout << TPOT / 1000.f << "ms/token";
        } else {
            std::cout << TPOT << "μs/token";
        }
        std::cout << " (";
        std::cout << (indices.size() - prompt_end) * 1.f / dur << " token/s), ";
        std::cout << context_used * 100 / rt.get_max_seq_len() << "% used ("
                  << context_used << " tokens)\n";
    }
    if (!mm.empty()) {
        logger.log(mm.generate_log());
    }
}

}
