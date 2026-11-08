#include "engine/attn.hpp"
#include "engine/utils.hpp"

#include "tensor/linalg.hpp"
#include "engine/runtime.hpp"
#include "util/utf8.hpp"
#include "util/ppm.hpp"
#include "util/fs.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace quetzal::mode {

static void viridis_color(float t, unsigned char* rgb) {
    static const unsigned char anchors[5][3] = {
        { 68,   1,  84},
        { 59,  82, 139},
        { 33, 145, 140},
        { 94, 201,  98},
        {253, 231,  37}
    };

    t = std::clamp(t, 0.f, 1.f) * 4.f;
    const std::size_t i = (std::min)(static_cast<std::size_t>(t), std::size_t(3));
    const float u = t - static_cast<float>(i);
    for (std::size_t c = 0; c < 3; ++c) {
        const float v = anchors[i][c] * (1.f - u) + anchors[i + 1][c] * u;
        rgb[c] = static_cast<unsigned char>(std::lround(v));
    }
}

static void write_attn_ppm(const std::string& path,
                           const tensor::tensor<float>& attn_mat,
                           bool enable_debug_info,
                           std::size_t scale = 6) {
    QUETZAL_ASSERT(attn_mat.shape().size() == 3, "attn_mat.shape().size() != 3");

    const std::size_t n_head = attn_mat.shape()[0];
    const std::size_t seq_len = attn_mat.shape()[2];
    const std::size_t height = seq_len * scale;
    const std::size_t width = n_head * (seq_len * scale + 1) - 1;

    util::ppm_writer pw(path, width, height, enable_debug_info);

    const float lo = -std::log10(static_cast<float>(seq_len) * 1000.f);
    for (std::size_t y = 0; y < height; ++y) {
        const std::size_t query = y / scale;
        for (std::size_t h = 0; h < n_head; ++h) {
            for (std::size_t x = 0; x < seq_len * scale; ++x) {
                const std::size_t key = x / scale;
                if (key > query) {
                    pw.write_pixel(90, 90, 90);
                    continue;
                }
                const float p = attn_mat[h][query][key];
                const float t = p > 0.f ? (std::log10(p) - lo) / (-lo) : 0.f;
                unsigned char rgb[3];
                viridis_color(t, rgb);
                pw.write_pixel(rgb[0], rgb[1], rgb[2]);
            }
            if (h + 1 < n_head) {
                pw.write_pixel(0, 0, 0);
            }
        }
    }
}

void attn_mode(const util::cli& cli) {
    runtime::runtime rt(cli);
    std::mt19937_64 gen(42);
    auto& model = rt.get_model();

    const std::string prompt = cli.get_system_prompt().empty()
        ? "你好！今天中午你准备吃什么？"
        : cli.get_system_prompt();
    auto indices = rt.get_tokenizer().encode(prompt);
    auto input = tensor::embedding_gather(model.get_tok_emb(), indices);

    auto transformer = model.get_blocks().front();
    auto attn = transformer.get_attention();
    input = tensor::layernorm(input,
                              transformer.get_layernorm1_weight(),
                              transformer.get_layernorm1_bias());
    auto attn_mat = attn.forward_attn(input);
    QUETZAL_ASSERT(attn_mat.shape().size() == 3, "attn_mat.shape().size() != 3");

    std::string output_dir = "mode.attn";
    if (!util::dir_exists(output_dir)) {
        util::create_dir(output_dir);
    }
    output_dir += util::is_windows() ? "\\" : "/";
    write_attn_ppm(output_dir + "attn.ppm", attn_mat, cli.enable_debug_info());

    std::uint32_t max_len = 0;
    for (auto i : indices) {
        max_len = (std::max)(max_len, utf8::utf8_str_wcwidth(rt.get_vocab(i)));
    }

    for (std::size_t i = 0; i < attn_mat.shape()[0]; ++i) {
        std::cout << "head [" << i << "]:" << std::endl;
        for (std::size_t j = 0; j < attn_mat.shape()[1]; ++j) {
            const auto& str = rt.get_vocab(indices[j]);
            const auto len = utf8::utf8_str_wcwidth(str);
            printf("\033[38;5;140m|%5u \033[0m\033[38;5;75m", indices[j]);
            utf8::print(std::cout, str);
            for (std::size_t k = 0; k < max_len - len; ++k) {
                std::cout << " ";
            }
            printf("\033[0m ");
            if ((j + 1) % 4 == 0) {
                std::cout << std::endl;
            }
        }
        if (indices.size() % 4 != 0) {
            std::cout << std::endl;
        }
        std::cout << "      ";
        for (std::size_t j = 0; j < attn_mat.shape()[1]; ++j) {
            printf("\033[38;5;140m%5u \033[0m", indices[j]);
        }
        std::cout << std::endl;
        for (std::size_t j = 0; j < attn_mat.shape()[1]; ++j) {
            printf("\033[38;5;140m%5u \033[0m", indices[j]);
            for (std::size_t k = 0; k < attn_mat.shape()[2]; ++k) {
                float item = attn_mat[i][j][k];
                if (0.25 <= item && item < 0.5) {
                    std::cout << "\033[38;5;51m";
                } else if (item >= 0.5) {
                    std::cout << "\033[48;5;51m";
                } else if (0.1 <= item  && item < 0.25) {
                    std::cout << "\033[38;5;162m";
                } else {
                    std::cout << "\033[48;5;162m";
                }
                printf("%5.2f \033[0m", item);
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

}