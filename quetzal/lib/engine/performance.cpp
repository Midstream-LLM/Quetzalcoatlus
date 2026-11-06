#include "engine/performance.hpp"
#include "engine/utils.hpp"
#include "engine/repetition_penalty.hpp"

#include "tensor/linalg.hpp"
#include "bbpe/tokenizer.hpp"
#include "model/qgpt_bin_reader.hpp"
#include "model/riverstrike/riverstrike.hpp"
#include "engine/runtime.hpp"
#include "util/chat_message.hpp"
#include "util/perf_info.hpp"
#include "util/timestamp.hpp"

#include <fstream>
#include <string>

namespace quetzal::mode {

void perf_mode(const util::cli& cli) {
    runtime::runtime rt(cli);
    std::mt19937_64 gen(42);
    auto& model = rt.get_model();

    std::cout << "[Info] performance mode\n";    

    util::message_manager mm(rt.get_tokenizer());
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

    std::string prompt = mm.build(rt.get_max_seq_len());
    std::vector<std::uint32_t> indices = rt.get_tokenizer().encode(prompt);
    std::size_t prompt_end = indices.size();
    std::size_t index = indices.back();
    util::timestamp ts;

    {
        ts.stamp();
        auto logits = model.prefill(indices);
        rt.get_penalty_applier().apply(indices, logits, prompt_end);
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
        rt.get_penalty_applier().apply(indices, logits, prompt_end);
        logits = logits / cli.get_temperature();
        tensor::apply_topk_mask(logits, cli.get_top_k());
        tensor::apply_topp_mask(logits, cli.get_top_p());
        auto topk = tensor::softmax<float>(logits);
        index = tensor::multinomial(topk, gen);
        indices.push_back(index);
        std::cout << "[Info] performance: warmup " << i + 1 << " cycle(s) "
                  << ts.elapsed_milli_seconds().count() << " ms\n";
    }

    std::string file = rt.get_model_config().model_name + "-" +
                       util::format_time(std::time(nullptr)) + ".perf.txt";
    std::ofstream perf_file_output(file);
    perf_file_output << "================================\n"
                     << "name: " << rt.get_model_config().model_name << "\n"
                     << "context: " << rt.get_max_seq_len() << "\n"
                     << "d_model: " << rt.get_model_config().d_model << "\n"
                     << "n_head: " << rt.get_model_config().n_head << "\n"
                     << "n_layer: " << rt.get_model_config().n_layer << "\n"
                     << "temperature: " << cli.get_temperature() << "\n"
                     << "topk: " << cli.get_top_k() << "\n"
                     << "topp: " << cli.get_top_p() << "\n"
                     << "repetition penalty: " << cli.get_repetition_penalty() << "\n"
                     << "frequency penalty: " << cli.get_frequency_penalty() << "\n"
                     << "presence penalty: " << cli.get_presence_penalty() << "\n";
    // perf for 20 cycles
    for (int i = 0; i < 20; ++i) {
        std::cout << "[Info] performance: test " << i + 1 << " cycle(s)\n";
        util::perf_info pi;
        pi.indices_length = indices.size();
        auto logits = model.decode(index, &pi);

        ts.stamp();
        rt.get_penalty_applier().apply(indices, logits, prompt_end);
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
    std::cout << "[Info] performance: reply \n"
              << rt.get_tokenizer().decode(indices) << "\n";
}

}
