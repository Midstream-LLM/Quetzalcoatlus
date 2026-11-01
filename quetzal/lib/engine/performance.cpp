#include "engine/performance.hpp"
#include "engine/utils.hpp"
#include "engine/repetition_penalty.hpp"

#include "tensor/linalg.hpp"
#include "bbpe/tokenizer.hpp"
#include "model/qgpt_bin_reader.hpp"
#include "model/riverstrike/riverstrike.hpp"
#include "util/chat_message.hpp"
#include "util/perf_info.hpp"
#include "util/timestamp.hpp"

#include <fstream>
#include <string>

namespace quetzal::mode {

void perf_mode(const util::cli& cli) {
    std::cout << "[Info] performance mode\n";
    gpt::qgpt_bin_reader qbr(cli.get_model_file_path(), cli.enable_debug_info());
    const auto& wm = qbr.get_weights_manager();
    const auto& br = qbr.get_bin_reader();
    bbpe::tokenizer tokenizer(br);
    repetition_penalty rp(cli.get_repetition_penalty(),
                          cli.get_frequency_penalty(),
                          cli.get_presence_penalty());

    const auto& cfg = qbr.get_model_config();
    gpt::riverstrike_jishui_v1 model(wm, cfg);
    qbr.release_memory();
    std::mt19937_64 gen(42);

    util::message_manager mm(tokenizer);
    if (!cli.system_prompt_disabled()) {
        mm.push("system",
            cli.get_system_prompt().empty()
                ? "你是一个聊天助手，请自然、友好地和用户聊天。"
                : cli.get_system_prompt()
        );
    }
    mm.push("user", "你好，今天感觉怎么样？");
    mm.push("assistant", "你好，今天感觉不错，你呢？");
    mm.push("user", "我也是。今天中午准备吃点什么？");
    mm.push("assistant", "今天中午准备吃烤牛排。");
    mm.push("user", "今天晚上准备吃点什么？");
    mm.push("assistant", "今天晚上准备吃烤鸡。");
    mm.push("user", "那吃的是相当不错了");

    std::string prompt = mm.build(cfg.max_seq_len);
    std::vector<std::uint32_t> indices = tokenizer.encode(prompt);
    std::size_t prompt_end = indices.size();
    std::size_t index = indices.back();
    util::timestamp ts;

    {
        ts.stamp();
        auto logits = model.prefill(indices);
        rp.apply(indices, logits, prompt_end);
        logits = logits / cli.get_temperature();
        tensor::apply_topk_mask(logits, cli.get_top_k());
        tensor::apply_topp_mask(logits, cli.get_top_p());
        auto topk = tensor::softmax<float>(logits);
        index = tensor::multinomial(topk, gen);
        indices.push_back(index);
        std::cout << "[Info] performance: prefill (TTFT) "
                  << ts.elapsed_milli_seconds().count() << " ms\n";
    }

    // warm up for 10 cycles
    for (int i = 0; i < 10; ++i) {
        ts.stamp();
        auto logits = model.decode(index);
        rp.apply(indices, logits, prompt_end);
        logits = logits / cli.get_temperature();
        tensor::apply_topk_mask(logits, cli.get_top_k());
        tensor::apply_topp_mask(logits, cli.get_top_p());
        auto topk = tensor::softmax<float>(logits);
        index = tensor::multinomial(topk, gen);
        indices.push_back(index);
        std::cout << "[Info] performance: warmup " << i + 1 << " cycle(s) "
                  << ts.elapsed_milli_seconds().count() << " ms\n";
    }

    std::string file = cfg.model_name + "-" +
                       util::format_time(std::time(nullptr)) + ".perf.txt";
    std::ofstream perf_file_output(file);
    // perf for 20 cycles
    for (int i = 0; i < 20; ++i) {
        std::cout << "[Info] performance: test " << i + 1 << " cycle(s)\n";
        util::perf_info pi;
        pi.indices_length = indices.size();
        auto logits = model.decode(index, &pi);

        ts.stamp();
        rp.apply(indices, logits, prompt_end);
        logits = logits / cli.get_temperature();
        tensor::apply_topk_mask(logits, cli.get_top_k());
        tensor::apply_topp_mask(logits, cli.get_top_p());
        auto topk = tensor::softmax<float>(logits);
        index = tensor::multinomial(topk, gen);
        pi.token_choose_perf = ts.elapsed_micro_seconds();
        indices.push_back(index);

        pi.dump(perf_file_output);
    }
    std::cout << "[Info] performance: written to " << file << "\n";
    std::cout << "[Info] performance: reply \n" << tokenizer.decode(indices) << "\n";
}

}
