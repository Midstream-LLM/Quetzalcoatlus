#include "engine/experiment.hpp"
#include "engine/utils.hpp"
#include "engine/repetition_penalty.hpp"

#include "tensor/linalg.hpp"
#include "bbpe/tokenizer.hpp"
#include "model/qgpt_bin_reader.hpp"
#include "model/riverstrike/riverstrike.hpp"
#include "util/chat_message.hpp"
#include "util/ppm.hpp"
#include "util/utf8.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace quetzal::mode {

static std::vector<float> band_scale(const std::vector<tensor::tensor<float>>& bands) {
    std::vector<float> scale(bands.size(), 1.f);
    for (std::size_t l = 0; l < bands.size(); ++l) {
        std::vector<float> mag(bands[l].total_size());
        for (std::size_t i = 0; i < mag.size(); ++i) {
            mag[i] = std::abs(bands[l].data()[i]);
        }
        const std::size_t k = static_cast<std::size_t>(0.995f * static_cast<float>(mag.size() - 1));
        std::nth_element(mag.begin(), mag.begin() + k, mag.end());
        scale[l] = mag[k] > 0.f ? mag[k] : 1.f;
    }
    return scale;
}

static void write_last_delta_ppm(const std::string& path,
                                 const std::vector<tensor::tensor<float>>& bands,
                                 std::size_t cols,
                                 std::size_t layer_cols,
                                 std::size_t cell,
                                 bool debug) {
    const std::size_t n_layer = bands.size();
    const std::size_t d_model = bands.front().shape()[1];
    QUETZAL_ASSERT(d_model % cols == 0, "[experiment] d_model is not divisible by cols");

    const std::size_t rows = d_model / cols;
    const std::size_t layer_rows = (n_layer + layer_cols - 1) / layer_cols;
    const std::size_t block_w = cols * cell;
    const std::size_t block_h = rows * cell;
    const std::size_t width = layer_cols * block_w + layer_cols - 1;
    const std::size_t height = layer_rows * block_h + layer_rows - 1;
    const std::vector<float> scale = band_scale(bands);

    util::ppm_writer pw(path, width, height);
    for (std::size_t py = 0; py < height; ++py) {
        const std::size_t layer_row = py / (block_h + 1);
        const std::size_t inner_y = py % (block_h + 1);
        for (std::size_t px = 0; px < width; ++px) {
            const std::size_t layer_col = px / (block_w + 1);
            const std::size_t inner_x = px % (block_w + 1);
            const std::size_t l = layer_row * layer_cols + layer_col;

            if (inner_y == block_h || inner_x == block_w || l >= n_layer) {
                pw.write_pixel(0, 0, 0);
                continue;
            }
            const auto& band = bands[l];
            const std::size_t last_row = band.shape()[0] - 1;
            const float v = band.data()[last_row * d_model
                                       + (inner_y / cell) * cols + inner_x / cell];
            pw.write_signed(v, scale[l]);
        }
    }
}

void experiment_mode(const util::cli& cli) {
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
    mm.push("user", "你好！介绍一下你自己。");

    info_dump(std::cout, cli, cfg, cfg.max_seq_len);

    std::string prompt = mm.build(cfg.max_seq_len);
    std::vector<std::uint32_t> indices = tokenizer.encode(prompt);
    std::size_t prompt_end = indices.size();
    std::vector<std::uint32_t> output_indices;

    const auto im_end = br.get_vocab_index().at("<|im_end|>");
    std::uint32_t index = 0;
    std::uint32_t count = 0;
    while (indices.size() < 70) {
        util::ppm_writer pw(
            "output." + std::to_string(count) + ".ppm",
            cfg.d_model,
            (indices.size() + 1) * cfg.n_layer,
            cli.enable_debug_info()
        );
        auto logits = model.forward_write_ppm(indices, pw);
        pw.write();
        write_last_delta_ppm(
            "last_delta." + std::to_string(count) + ".ppm",
            pw.bands(), 22, 6, 3,
            cli.enable_debug_info()
        );

        rp.apply(indices, logits, prompt_end);
        logits = logits / cli.get_temperature();
        tensor::apply_topk_mask(logits, cli.get_top_k());
        tensor::apply_topp_mask(logits, cli.get_top_p());
        auto topk = tensor::softmax<float>(logits);
        index = tensor::multinomial<float>(topk, gen);
        visualize_topk(br.get_vocab(), topk, cli.get_top_k(), index);
        if (index == im_end) {
            break;
        }
        indices.push_back(index);
        output_indices.push_back(index);
        ++count;
    }
    std::cout << std::endl;
    std::cout << "[Quetzal] " << tokenizer.decode(output_indices) << std::endl;
}

}
