#pragma once

#include "tensor/tensor.hpp"

#include <vector>
#include <cstdint>

namespace quetzal::mode {

class repetition_penalty {
private:
    float repetition_penalty_ = 1.f;
    float frequency_penalty_ = 0.f;
    float presence_penalty_ = 0.f;

public:
    repetition_penalty(float repetition_penalty_param,
                       float frequency_penalty_param,
                       float presence_penalty_param) :
        repetition_penalty_(repetition_penalty_param),
        frequency_penalty_(frequency_penalty_param),
        presence_penalty_(presence_penalty_param) {}

    void apply(const std::vector<std::uint32_t>& indices,
               tensor::tensor<float>& logits,
               std::size_t prompt_end) const;
};

}
