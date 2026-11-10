#include "bbpe/tokenizer.hpp"
#include "util/utf8.hpp"
#include "util/byte_to_unicode.hpp"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>

namespace quetzal::bbpe {

void tokenizer::init(const std::vector<std::string>& special) {
    for (std::uint32_t i = 0; i < 256; i++) {
        auto c = static_cast<char>(i);
        vocab_index.insert(std::string(1, c), vocab_index.size());
        vocab.push_back(std::string(1, c));
    }

    for (auto i : special) {
        vocab_index.insert(i, vocab_index.size());
        vocab.push_back(i);
    }
}

void tokenizer::read_to_text(std::string& text, std::istream& ifs) {
    while (ifs) {
        std::string line;
        std::getline(ifs, line);

        if (line.empty()) {
            continue;
        }

        text += line;
    }
}

void tokenizer::replace(std::vector<std::uint32_t>& src, index_pair p) {
    auto new_str = vocab[p.left] + vocab[p.right];
    auto id = vocab_index.at(new_str);

    std::size_t write = 0;
    for (std::size_t read = 0; read < src.size(); read ++) {
        if (src[read] == p.left && read + 1 < src.size() && src[read + 1] == p.right) {
            src[write ++] = id;
            read ++;
        } else {
            src[write ++] = src[read];
        }
    }

    src.resize(write);
}

bool tokenizer::single_merge(std::vector<std::uint32_t>& src) {
    util::densemap<index_pair, std::uint32_t> merge;
    for (std::size_t i = 0; i < src.size(); i++) {
        if (i + 1 < src.size()) {
            auto key = index_pair(src[i], src[i + 1]);
            if (merge.contains(key)) {
                merge[key]++;
            } else {
                merge.insert(key, 1);
            }
        }
    }

    std::uint32_t max_count = 0;
    for (const auto& [key, value] : merge) {
        if (value > max_count) {
            max_count = value;
        }
    }

    if (max_count < 3) {
        return false;
    }

    for (const auto& [key, value] : merge) {
        if (value != max_count) {
            continue;
        }

        if (pair_index.contains(key)) {
            continue;
        }

        auto new_str = vocab[key.left] + vocab[key.right];
        if (new_str.size() > 32) {
            std::cerr << "[BBPE-Warning] too long word: " << new_str << "\n";
        }

        vocab_index.insert(new_str, vocab_index.size());
        vocab.push_back(new_str);

        pair_index.insert(key, pair_index.size());
        merge_pairs.push_back(key);

        replace(src, key);
        return true;
    }

    return false;
}

tokenizer::tokenizer(const std::vector<std::string>& special) {
    init(special);
    special_vocab = special;
    special_vocab_size = special.size();
}

tokenizer::tokenizer(const gpt::bin_reader& br) {
    vocab_index = br.get_vocab_index();
    vocab = br.get_vocab();
    merge_pairs = br.get_merges();

    special_vocab = br.get_special_vocab();
    special_vocab_size = special_vocab.size();
}

bool tokenizer::merge(const std::string& path, std::uint32_t max_vocab_size) {
    std::ifstream ifs(path);
    if (!ifs) {
        std::cerr << "[Info] tokenizer: cannot open <" << path << ">" << std::endl;
        return false;
    }

    std::string text;
    read_to_text(text, ifs);

    auto src = encode(text);
    auto prev = src.size();

    std::cout << "[Info] init encoded: " << src.size() << "\n";
    std::cout << "[Info] text: ";
    if (text.length() / 1024.0 < 1.0) {
        std::cout << text.length() << "Byte\n";
    } else if (text.length() / 1024.0 / 1024.0 < 1.0) {
        std::cout << text.length() / 1024.0 << " KB\n";
    } else {
        std::cout << text.length() / 1024.0 / 1024.0 << "M\n";
    }

    std::uint64_t count = 0;

    auto start = std::chrono::high_resolution_clock::now();
    while (single_merge(src)) {
        count ++;
        if (count % 5 == 0) {
            auto end = std::chrono::high_resolution_clock::now();
            auto sec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
            auto diff = prev - src.size();
            std::cout << "[Info] diff: " << diff;
            std::cout << " (" << diff / static_cast<double>(src.size()) * 100.0 << "%)";
            std::cout << " src: " << src.size() << "(" << src.size() * sizeof(std::uint32_t) / 1024.0 / 1024.0 << "MB)";
            std::cout << " vocab: " << vocab.size();
            std::cout << " speed: " << 5.0 / sec << " iter/s\n";
            prev = src.size();
            start = end;
        }

        if (vocab.size() >= max_vocab_size) {
            break;
        }
    }

    std::cout << "[Info] final encoded: " << src.size() << "\n";
    std::cout << "[Info] final vocab  : " << vocab.size() << "\n";
    std::cout << "[Info] final merge  : " << merge_pairs.size() << "\n";
    return true;
}

void tokenizer::dump(std::ostream& os) const {
    std::vector<std::string> table;
    table.resize(vocab.size());
    for (const auto& [key, value] : vocab_index) {
        table[value] = key;
    }

    util::converter cvt;

    for (const auto& key : table) {
        os << "[" << vocab_index.at(key) << "] ";
        utf8::print(os, key) << "\n";

        auto encoded = cvt.encode(key);
        os << "  encoded: " << encoded << "\n";

        auto decoded = cvt.decode(encoded);
        os << "  decoded: ";
        utf8::print(os, decoded) << "\n";
    }
}

void tokenizer::dump_json(std::ostream& os) const {
    auto raw = [&](const std::string& s) {
        for (auto c : s) {
            if (c == '\"') {
                os << "\\\"";
            } else if (c == '\\') {
                os << "\\\\";
            } else {
                os << c;
            }
        }
    };

    util::converter cvt;

    os << "{\n";
    os << "  \"version\": \"1.0\",\n";
    os << "  \"truncation\": null,\n";
    os << "  \"padding\": null,\n";
    os << "  \"normalizer\": null,\n";
    os << "  \"pre_tokenizer\": {\n";
    os << "    \"type\": \"ByteLevel\",\n";
    os << "    \"add_prefix_space\": false,\n";
    os << "    \"trim_offsets\": true,\n";
    os << "    \"use_regex\": true\n";
    os << "  },\n";
    os << "  \"decoder\": {\n";
    os << "    \"type\": \"ByteLevel\",\n";
    os << "    \"add_prefix_space\": true,\n";
    os << "    \"trim_offsets\": true,\n";
    os << "    \"use_regex\": true\n";
    os << "  },\n";
    os << "  \"post_processor\": {\n";
    os << "    \"type\": \"TemplateProcessing\",\n";
    os << "    \"single\": [\n";
    os << "      {\n";
    os << "        \"Sequence\": {\n";
    os << "          \"id\": \"A\",\n";
    os << "          \"type_id\": 0\n";
    os << "        }\n";
    os << "      }\n";
    os << "    ],\n";
    os << "    \"pair\": [\n";
    os << "      {\n";
    os << "        \"Sequence\": {\n";
    os << "          \"id\": \"A\",\n";
    os << "          \"type_id\": 0\n";
    os << "        }\n";
    os << "      },\n";
    os << "      {\n";
    os << "        \"Sequence\": {\n";
    os << "          \"id\": \"B\",\n";
    os << "          \"type_id\": 1\n";
    os << "        }\n";
    os << "      }\n";
    os << "    ],\n";
    os << "    \"special_tokens\": {}\n";
    os << "  },\n";
    os << "  \"added_tokens\": [\n";
    for (const auto& i : special_vocab) {
        os << "    {\n";
        os << "      \"id\": " << vocab_index.at(i) << ",\n";
        os << "      \"content\": \"";
        raw(i);
        os << "\",\n";
        os << "      \"single_word\": false,\n";
        os << "      \"lstrip\": false,\n";
        os << "      \"rstrip\": false,\n";
        os << "      \"normalized\": false,\n";
        os << "      \"special\": true\n";
        os << "    }";
        if (i != special_vocab.back()) {
            os << ",";
        }
        os << "\n";
    }
    os << "  ],\n";
    os << "  \"model\": {\n";
    os << "    \"type\": \"BPE\",\n";
    os << "    \"dropout\": null,\n";
    os << "    \"unk_token\": \"<|unk|>\",\n";
    os << "    \"continuing_subword_prefix\": null,\n";
    os << "    \"end_of_word_suffix\": null,\n";
    os << "    \"fuse_unk\": false,\n";
    os << "    \"byte_fallback\": false,\n";
    os << "    \"ignore_merges\": false,\n";
    os << "    \"vocab\": {\n";
    for (const auto& i : vocab) {
        os << "      \"";

        auto encoded = cvt.encode(i);
        raw(encoded);

        os << "\": " << vocab_index.at(i);
        if (i != vocab.back()) {
            os << ",";
        }
        os << "\n";
    }
    os << "    },\n";
    os << "    \"merges\": [\n";
    for (const auto& p : merge_pairs) {
        auto l = cvt.encode(vocab[p.left]);
        auto r = cvt.encode(vocab[p.right]);
        os << "      [ \"";
        raw(l);
        os << "\", \"";
        raw(r);
        os << "\" ]";
        if (p != merge_pairs.back()) {
            os << ",";
        }
        os << "\n";
    }
    os << "    ]\n";
    os << "  }\n";
    os << "}\n";
}

std::vector<std::uint32_t> tokenizer::encode(const std::string& text) const {
    std::vector<std::uint32_t> result;
    std::vector<std::string> special_tokens;
    for (auto i : special_vocab) {
        special_tokens.push_back(i);
    }
    std::sort(special_tokens.begin(), special_tokens.end(), [&](const std::string& a, const std::string& b) {
        return a.length() > b.length();
    });

    for (std::size_t i = 0; i < text.length(); i++) {
        auto c = text[i];
        if (c == '<') {
            bool found = false;
            for (const auto& token : special_tokens) {
                if (text.compare(i, token.length(), token) == 0) {
                    result.push_back(vocab_index.at(token));
                    i += token.length() - 1;
                    found = true;
                    break;
                }
            }
            if (found) {
                continue;
            }
        }
        result.push_back(vocab_index.at(std::string(1, c)));
    }

    if (merge_pairs.empty()) {
        return result;
    }

    for (std::size_t i = 0; i < merge_pairs.size(); i++) {
        auto [left, right] = merge_pairs[i];
        auto new_id = 256 + special_vocab_size + i;
        for (std::size_t pos = 0; pos + 1 < result.size(); ) {
            if (result[pos] == left && result[pos + 1] == right) {
                result[pos] = new_id;
                result.erase(result.begin() + pos + 1);
            } else {
                pos++;
            }
        }
    }

    return result;
}

std::string tokenizer::decode(const std::vector<std::uint32_t>& indices) const {
    std::string result;
    for (auto i : indices) {
        if (i >= vocab.size()) {
            throw std::out_of_range("tokenizer: index out of range: " +
                                    std::to_string(i) + " >= " +
                                    std::to_string(vocab.size()));
        }
        result += vocab.at(i);
    }
    return result;
}

}
