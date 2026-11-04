#pragma once

#include "bbpe/tokenizer.hpp"
#include "model/riverstrike/riverstrike.hpp"
#include "model/qgpt_bin_reader.hpp"
#include "tensor/scaling.hpp"
#include "engine/repetition_penalty.hpp"
#include "util/cli.hpp"

#include <optional>

namespace quetzal::runtime {

class runtime {
private:
    gpt::qgpt_bin_reader qbr;
    bbpe::tokenizer tokenizer;
    std::size_t real_max_seq_len;
    std::optional<tensor::NTK_aware_scaling> rule;
    gpt::riverstrike_jishui_v1 model;
    mode::repetition_penalty penalty;

public:
    explicit runtime(const util::cli& cli);
    runtime(const runtime&) = delete;
    runtime& operator=(const runtime&) = delete;
    std::size_t get_max_seq_len() const { return real_max_seq_len; }
    const auto& get_vocab(std::size_t index) const {
        return qbr.get_bin_reader().get_vocab().at(index);
    }
    const auto& get_vocab_table() const { return qbr.get_bin_reader().get_vocab(); }
    auto get_vocab_index(const std::string& word) const {
        return qbr.get_bin_reader().get_vocab_index().at(word);
    }
    const auto& get_tokenizer() const { return tokenizer; }
    const auto& get_model_config() const { return qbr.get_model_config(); }
    auto& get_model() { return model; }
    const auto& get_penalty_applier() const { return penalty; }
};

}
