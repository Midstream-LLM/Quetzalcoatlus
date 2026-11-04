#include "engine/experiment.hpp"
#include "engine/utils.hpp"
#include "engine/repetition_penalty.hpp"

#include "tensor/linalg.hpp"
#include "bbpe/tokenizer.hpp"
#include "model/qgpt_bin_reader.hpp"
#include "model/riverstrike/riverstrike.hpp"
#include "engine/runtime.hpp"
#include "util/chat_message.hpp"
#include "util/ppm.hpp"
#include "util/utf8.hpp"
#include "util/fs.hpp"

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
    runtime::runtime rt(cli);
    std::mt19937_64 gen(42);
    auto& model = rt.get_model();

    util::message_manager mm(rt.get_tokenizer());
    if (!cli.system_prompt_disabled()) {
        mm.push("system",
            cli.get_system_prompt().empty()
                ? "你是一个聊天助手，请自然、友好地和用户聊天。"
                : cli.get_system_prompt()
        );
    }
    mm.push("user", "你好！介绍一下你自己。");

    info_dump(std::cout, cli, rt.get_model_config(), rt.get_max_seq_len());

    std::string prompt = mm.build(rt.get_max_seq_len());
    std::vector<std::uint32_t> indices = rt.get_tokenizer().encode(prompt);
    std::size_t prompt_end = indices.size();
    std::vector<std::uint32_t> output_indices;

    std::string output_dir = "mode.expr";
    if (!util::dir_exists(output_dir)) {
        util::create_dir(output_dir);
    }
    output_dir += util::is_windows() ? "\\" : "/";

    const auto im_end = rt.get_vocab_index("<|im_end|>");
    std::uint32_t index = 0;
    std::uint32_t count = 0;
    while (indices.size() < 70) {
        util::ppm_writer pw(
            output_dir + "output." + std::to_string(count) + ".ppm",
            rt.get_model_config().d_model,
            (indices.size() + 1) * rt.get_model_config().n_layer,
            cli.enable_debug_info()
        );
        auto logits = model.forward_write_ppm(indices, pw);
        pw.write();
        write_last_delta_ppm(
            output_dir + "last_delta." + std::to_string(count) + ".ppm",
            pw.bands(), 22, 6, 3,
            cli.enable_debug_info()
        );

        rt.get_penalty_applier().apply(indices, logits, prompt_end);
        logits = logits / cli.get_temperature();
        tensor::apply_topk_mask(logits, cli.get_top_k());
        tensor::apply_topp_mask(logits, cli.get_top_p());
        auto topk = tensor::softmax<float>(logits);
        index = tensor::multinomial<float>(topk, gen);
        visualize_topk(rt.get_vocab_table(), topk, cli.get_top_k(), index);
        if (index == im_end) {
            break;
        }
        indices.push_back(index);
        output_indices.push_back(index);
        ++count;
    }
    std::cout << std::endl;
    std::cout << "[Quetzal] "
              << rt.get_tokenizer().decode(output_indices) << std::endl;
}

}
