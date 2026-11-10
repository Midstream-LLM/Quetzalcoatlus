#include "bbpe/tokenizer.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>

void test(const quetzal::bbpe::tokenizer& bpe) {
    const std::vector<std::string> test_cases = {
        "5 年，你知道我这 5 年都是怎么过的吗",
        "我去不早说",
        "事情终于有了新的退展",
        "<|im_start|><|im_end|><|pad|><|unk|><|endoftext|>",
        "<|im_start|>你好<|im_end|>hello？<|pad|>"
    };

    for (const auto& text : test_cases) {
        auto input = bpe.encode(text);
        auto output = bpe.decode(input);

        std::cout << "[TEST] Input: " << text << "\n";
        std::cout << "[TEST] Indices: " << "[";
        for (auto i : input) {
            std::cout << i << " ";
        }
        std::cout << "]\n";
        std::cout << "[TEST] Output: " << bpe.decode(bpe.encode(text)) << "\n";
        assert(text == output);
    }

    std::cout << "[TEST] All tests passed\n";
}

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input text>\n";
        return 1;
    }

    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        std::cout << "Usage: " << argv[0] << " <input text>\n";
        return 0;
    }

    const std::vector<std::string> special = {
        "<|pad|>", "<|unk|>", "<|endoftext|>",
        "<|im_start|>", "<|im_end|>"
    };

    quetzal::bbpe::tokenizer bpe(special);
    if (!bpe.merge(argv[1], 12800)) {
        return 1;
    }

    test(bpe);

    std::ofstream log_out("data/tokenizer.log");
    bpe.dump(log_out);

    std::ofstream json_out("data/tokenizer.json");
    bpe.dump_json(json_out);
    return 0;
}
