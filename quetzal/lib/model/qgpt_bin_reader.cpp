#include "model/qgpt_bin_reader.hpp"

namespace quetzal::gpt {

static std::uint32_t read_u32(std::istream& in) {
    std::uint32_t v = 0;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!in) {
        throw std::runtime_error("unexpected EOF");
    }
    return v;
}

void weights_manager::read(std::ifstream& in, bool debug_mode) {
    const std::uint32_t tensor_count = read_u32(in);

    for (std::uint32_t i = 0; i < tensor_count; ++i) {
        if (debug_mode) {
            std::cout << "[Debug] weights_manager: loading tensor "
                      << i + 1 << "/" << tensor_count
                      << " (" << int((i + 1) * 100.f / tensor_count) << "%)\r"
                      << std::flush;
        }
        // read tensor name
        std::string name;
        const std::uint32_t name_len = read_u32(in);
        name.resize(name_len);
        in.read(name.data(), static_cast<std::streamsize>(name_len));
        if (!in) {
            throw std::runtime_error("truncated name");
        }

        // read shape
        std::vector<std::size_t> shape;
        const std::uint32_t ndim = read_u32(in);
        shape.resize(ndim);
        std::size_t numel = 1;
        for (std::uint32_t d = 0; d < ndim; ++d) {
            shape[d] = read_u32(in);
            numel *= shape[d];
        }

        // read dtype enum
        const std::uint32_t dtype = read_u32(in);
        if (static_cast<weight_dtype>(dtype) != weight_dtype::float32) {
            throw std::runtime_error("unsupported dtype " + std::to_string(dtype) +
                                     " for tensor " + name);
        }

        tensor::tensor<float> t(shape);

        // read tensor buffer
        in.read(reinterpret_cast<char*>(t.data()),
                static_cast<std::streamsize>(numel * sizeof(float)));
        if (!in) {
            throw std::runtime_error("truncated data for " + name);
        }

        weights_.emplace(name, std::move(t));
    }

    if (debug_mode) {
        std::cout << std::endl;
        auto parameters = total_parameters();
        std::cout << "[Debug] weights_manager: total parameters "
                  << parameters << " (" << parameters / 1000000 << " M)\n";
        std::cout << "[Debug] weights_manager: parameters memory usage "
                  << parameters * sizeof(float) / (1024 * 1024) << " MB\n";
    }
}

void bin_reader::read(std::ifstream& in, bool debug_mode) {
    // read added token list
    auto added_tokens_len = read_u32(in);
    for (std::uint32_t i = 0; i < added_tokens_len; ++i) {
        auto token_len = read_u32(in);
        std::string token;
        token.resize(token_len);
        in.read(token.data(), static_cast<std::streamsize>(token_len));
        if (!in) {
            throw std::runtime_error("truncated added token");
        }

        special_vocab.push_back(token);
    }
    if (debug_mode) {
        std::cout << "[Debug] bin_reader: special vocab " << special_vocab.size() << "\n";
    }

    // read vocab
    auto vocab_size = read_u32(in);
    for (std::uint32_t i = 0; i < vocab_size; ++i) {
        auto token_index = read_u32(in);
        auto token_len = read_u32(in);
        std::string token;
        token.resize(token_len);
        in.read(token.data(), static_cast<std::streamsize>(token_len));
        if (!in) {
            throw std::runtime_error("truncated token");
        }

        vocab.push_back(token);
        vocab_index.insert(token, token_index);
    }
    if (debug_mode) {
        std::cout << "[Debug] bin_reader: vocab " << vocab.size() << "\n";
    }

    // read merges
    auto merges_size = read_u32(in);
    for (std::uint32_t i = 0; i < merges_size; ++i) {
        auto token_a_len = read_u32(in);
        std::string token_a;
        token_a.resize(token_a_len);
        in.read(token_a.data(), static_cast<std::streamsize>(token_a_len));
        if (!in) {
            throw std::runtime_error("truncated token");
        }

        auto token_b_len = read_u32(in);
        std::string token_b;
        token_b.resize(token_b_len);
        in.read(token_b.data(), static_cast<std::streamsize>(token_b_len));
        if (!in) {
            throw std::runtime_error("truncated token");
        }

        merge_pairs.push_back(bbpe::index_pair {
            vocab_index.at(token_a),
            vocab_index.at(token_b)
        });
    }
    if (debug_mode) {
        std::cout << "[Debug] bin_reader: merges " << merge_pairs.size() << "\n";
    }
}

void qgpt_bin_reader::read_config(std::ifstream& in, bool debug_mode) {
    util::densemap<std::string, std::uint32_t> num_values;
    util::densemap<std::string, std::string> str_values;

    auto total_configs = read_u32(in);
    for (std::uint32_t i = 0; i < total_configs; ++i) {
        auto str_len = read_u32(in);
        std::string key;
        key.resize(str_len);
        in.read(key.data(), static_cast<std::streamsize>(str_len));

        auto type = static_cast<value_type>(read_u32(in));
        switch (type) {
            case value_type::uint32: {
                auto value = read_u32(in);
                num_values.insert(key, value);
                if (debug_mode) {
                    std::cout << "[Debug] qgpt_bin_reader: read model config "
                              << "'" << key << "': " << value << std::endl;
                }
            } break;
            case value_type::string: {
                auto value_len = read_u32(in);
                std::string value;
                value.resize(value_len);
                in.read(value.data(), static_cast<std::streamsize>(value_len));
                str_values.insert(key, value);
                if (debug_mode) {
                    std::cout << "[Debug] qgpt_bin_reader: read model config "
                              << "'" << key << "': \""
                              << value << "\"" << std::endl;
                }
            } break;
            default:
                throw std::invalid_argument("unknown config value type");
        }
    }

    if (str_values.contains("model_name")) {
        model_config_.model_name = str_values.at("model_name");
    } else {
        throw std::invalid_argument("'model_name' not found");
    }

    if (str_values.contains("default_system_prompt")) {
        model_config_.default_system_prompt = str_values.at("default_system_prompt");
    } else {
        model_config_.default_system_prompt = "You are a helpful assistant.";
    }

    if (num_values.contains("d_model")) {
        model_config_.d_model = num_values.at("d_model");
    } else {
        throw std::invalid_argument("'d_model' not found");
    }

    if (num_values.contains("n_head")) {
        model_config_.n_head = num_values.at("n_head");
    } else {
        throw std::invalid_argument("'n_head' not found");
    }

    if (num_values.contains("n_layer")) {
        model_config_.n_layer = num_values.at("n_layer");
    } else {
        throw std::invalid_argument("'n_layer' not found");
    }

    if (num_values.contains("max_seq_len")) {
        model_config_.max_seq_len = num_values.at("max_seq_len");
    } else {
        throw std::invalid_argument("'max_seq_len' not found");
    }
}

qgpt_bin_reader::qgpt_bin_reader(const std::string& path, bool debug_mode) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }

    char magic[4];
    in.read(magic, 4);
    if (!in || std::string(magic, 4) != "QGPT") {
        throw std::runtime_error("not a QGPT model file: " + path);
    }

    read_config(in, debug_mode);
    weights_manager_.read(in, debug_mode);
    bin_reader_.read(in, debug_mode);
}

}
