#pragma once

#include <string>

namespace quetzal::gpt {

struct model_config {
    std::string model_name;
    std::size_t d_model = 0;
    std::size_t n_head = 0;
    std::size_t n_layer = 0;
    std::size_t max_seq_len = 0;
    std::string default_system_prompt;
};

}
