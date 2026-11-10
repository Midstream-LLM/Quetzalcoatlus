#pragma once

#include "util/densemap.hpp"
#include "bbpe/index_pair.hpp"
#include "model/qgpt_bin_reader.hpp"

#include <string>
#include <cstdint>
#include <vector>
#include <iostream>

namespace quetzal::bbpe {

class tokenizer {
private:
    util::densemap<std::string, std::uint32_t> vocab_index;
    std::vector<std::string> vocab;

    std::vector<std::string> special_vocab;
    std::uint32_t special_vocab_size;

    util::densemap<index_pair, std::uint32_t> pair_index;
    std::vector<index_pair> merge_pairs;

private:
    void init(const std::vector<std::string>& special);
    void read_to_text(std::string& text, std::istream& ifs);
    void replace(std::vector<std::uint32_t>&, index_pair);
    bool single_merge(std::vector<std::uint32_t>&);

public:
    tokenizer(const std::vector<std::string>& special);
    tokenizer(const gpt::bin_reader& br);
    bool merge(const std::string& path, std::uint32_t max_vocab_size);
    void dump(std::ostream&) const;
    void dump_json(std::ostream&) const;
    std::vector<std::uint32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<std::uint32_t>& indices) const;
};

}
