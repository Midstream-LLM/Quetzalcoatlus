#include "engine/repetition_penalty.hpp"

#include <unordered_map>

namespace quetzal::mode {

void repetition_penalty::apply(const std::vector<std::uint32_t>& indices,
                               tensor::tensor<float>& logits,
                               std::size_t prompt_end) const {
    const bool multiplicative = repetition_penalty_ > 1.0f;
    const bool additive = presence_penalty_ != 0.f || frequency_penalty_ != 0.f;
    if ((!multiplicative && !additive) || indices.size() <= prompt_end) {
        return;
    }
    std::unordered_map<std::uint32_t, std::size_t> seen;
    for (std::size_t i = prompt_end; i < indices.size(); ++i) {
        if (!seen.count(indices[i])) {
            seen.emplace(indices[i], 1);
            continue;
        }
        seen[indices[i]]++;
    }
    for (const auto& [token, n] : seen) {
        auto& x = logits.data()[token];
        x = (x > 0) ? x / repetition_penalty_ : x * repetition_penalty_;
        x -= presence_penalty_ + frequency_penalty_ * static_cast<float>(n);
    }
}

}
