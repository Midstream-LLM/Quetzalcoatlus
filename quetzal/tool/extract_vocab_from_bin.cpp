#include "model/qgpt_bin_reader.hpp"
#include "bbpe/tokenizer.hpp"
#include "util/timestamp.hpp"

#include <iostream>
#include <fstream>
#include <string>

/* If you find 'oops i deleted the vocab file without any backup :(',
 * but you still have an exported model bin file, you can use this
 * program to extract the vocab from the model file.
 *                                   - And don't ask me why i have this idea :)
 */
int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <model_file>" << std::endl;
        return 1;
    }

    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        std::cout << "Usage: " << argv[0] << " <model_file>\n";
        return 0;
    }

    try {
        quetzal::gpt::qgpt_bin_reader qbr(argv[1], true);
        quetzal::bbpe::tokenizer tk(qbr.get_bin_reader());

        std::string file = "tokenizer." +
            quetzal::util::format_time(std::time(nullptr));
        std::cout << "[Info] extracting vocab to " << file << ".json\n";
        {
            std::ofstream os(file + ".json");
            tk.dump_json(os);
        }
        std::cout << "[Info] extracting vocab to " << file << ".log\n";
        {
            std::ofstream os(file + ".log");
            tk.dump(os);
        }
    } catch (std::exception &e) {
        std::cout << "[Error] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}