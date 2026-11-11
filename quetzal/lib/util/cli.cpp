#include "tensor/omp.hpp"
#include "util/cli.hpp"

#include <vector>
#include <cstring>
#include <cmath>
#include <cerrno>
#include <cstdlib>
#include <charconv>

namespace quetzal::util {

static bool streq(const char* a, const char* b) {
    return strcmp(a, b) == 0;
}

float cli::parse_float(const char* arg, const char* value) const {
    char* end = nullptr;
    errno = 0;
    float res = std::strtof(value, &end);
    if (end == value || *end != '\0' || errno == ERANGE || !std::isfinite(res)) {
        report_invalid_argument(arg, value);
    }
    return res;
}

int cli::parse_int(const char* arg, const char* value) const {
    int res = 0;
    const char* end = value + std::strlen(value);
    auto [ptr, ec] = std::from_chars(value, end, res);
    if (ec != std::errc() || ptr != end) {
        report_invalid_argument(arg, value);
    }
    return res;
}

void cli::help(std::ostream& out) const {
    out << "Usage: " << executable_name
        << " <model_file_path>" << std::endl;
    out << "Options:" << std::endl;
    out << "  -h, --help                        "
        << "| print help message" << std::endl;
    out << "  -v, --version                     "
        << "| print version" << std::endl;
    out << "      --debug                       "
        << "| enable debug info" << std::endl;
    out << "      --disable-system-prompt       "
        << "| disable system prompt" << std::endl;
    out << "Modes:" << std::endl;
    out << "  -c, --chat                        "
        << "| enable chat mode (default)" << std::endl;
    out << "  -e, --expr                        "
        << "| enable experimental mode" << std::endl;
    out << "  -a, --attn                        "
        << "| enable attn visualization mode" << std::endl;
    out << "  -p, --perf                        "
        << "| enable performance debug mode" << std::endl;
    out << "Parameters:" << std::endl;
    out << "  -s, --system-prompt <string>      "
        << "| set system prompt (default: model default setting)" << std::endl;
    out << "      --top-k <int>                 "
        << "| set top-k (default: 20)" << std::endl;
    out << "      --top-p <float>               "
        << "| set top-p (default: 0.9)" << std::endl;
    out << "      --temperature <float>         "
        << "| set temperature (default: 0.8)" << std::endl;
    out << "      --repetition-penalty <float>  "
        << "| set repetition penalty (default: 1.15, 1.0~ is valid)" << std::endl;
    out << "      --frequency-penalty <float>   "
        << "| set frequency penalty (default: 0.0, -2.0~2.0 is valid)" << std::endl;
    out << "      --presence-penalty <float>    "
        << "| set presence penalty (default: 0.0, -2.0~2.0 is valid)" << std::endl;
    out << "      --seed <int>                  "
        << "| set random seed (default std::random_device)" << std::endl;
    out << "      --rope-scaling-factor <float> "
        << "| set rope scaling factor (default: 1, 1.0~8.0 is valid)" << std::endl;
}

void cli::print_version(std::ostream& out) const {
    out << "Quetzalcoatlus Inference Engine v" << version()
        << " (OpenMP " << (tensor::omp_enabled ? "enabled" : "disabled")
        << ")" << std::endl;
}

const char* cli::version() const {
    // version 1.0.0 we finished the inference engine, with simple kv-cache
    return "1.0.2";
}

void cli::logo(std::ostream& os) const {
    std::string logo_content = "\n"
    "      __   _  _  ____  ____  ____   __   __           \n"
    "     /  \\ / )( \\(  __)(_  _)(__  ) / _\\ (  )       \n"
    "    (  O )) \\/ ( ) _)   )(   / _/ /    \\/ (_/\\     \n"
    "     \\__\\)\\____/(____) (__) (____)\\_/\\_/\\____/  \n"
    "      ___  __    __  ____  __    _  _  ____           \n"
    "     / __)/  \\  / _\\(_  _)(  )  / )( \\/ ___)       \n"
    "    ( (__(  O )/    \\ )(  / (_/\\) \\/ (\\___ \\     \n"
    "     \\___)\\__/ \\_/\\_/(__) \\____/\\____/(____/  \n\n"
    "            Presented by Midstream-AI                 \n"
    "            ~~~~~~~~~~~~~~~~~~~~~~~~~               \n\n";

    os << "\033[38;5;51m" << logo_content << "\033[0m";
}

cli::cli(int argc, const char** argv) {
    executable_name = argv[0];
    if (argc < 2) {
        report_and_exit(std::cerr, 1);
    }
    if (streq(argv[1], "--help") || streq(argv[1], "-h")) {
        report_and_exit(std::cout, 0);
    } else if (streq(argv[1], "--version") || streq(argv[1], "-v")) {
        print_version(std::cout);
        std::exit(0);
    }
    model_file_path = argv[1];

    for (int i = 2; i < argc; i++) {
        if (streq(argv[i], "--help") || streq(argv[i], "-h")) {
            report_and_exit(std::cout, 0);
        } else if (streq(argv[i], "--version") || streq(argv[i], "-v")) {
            print_version(std::cout);
            std::exit(0);
        }

        if (streq(argv[i], "--chat") || streq(argv[i], "-c")) {
            run_mode = mode::chat;
        } else if (streq(argv[i], "--expr") || streq(argv[i], "-e")) {
            run_mode = mode::experimental;
        } else if (streq(argv[i], "--attn") || streq(argv[i], "-a")) {
            run_mode = mode::attn;
        } else if (streq(argv[i], "--perf") || streq(argv[i], "-p")) {
            run_mode = mode::perf;
        } else if (streq(argv[i], "--debug")) {
            show_debug_info = true;
        } else if (streq(argv[i], "--disable-system-prompt")) {
            disable_system_prompt = true;
        } else if (streq(argv[i], "-s") || streq(argv[i], "--system-prompt")) {
            require_value(i, argc, argv[i]);
            system_prompt = argv[i + 1];
            i++;
        } else if (streq(argv[i], "--top-k")) {
            require_value(i, argc, argv[i]);
            int res = parse_int(argv[i], argv[i + 1]);
            if (res <= 0) {
                report_invalid_argument(argv[i], argv[i + 1]);
            }
            top_k = res;
            i++;
        } else if (streq(argv[i], "--top-p")) {
            require_value(i, argc, argv[i]);
            top_p = parse_float(argv[i], argv[i + 1]);
            if (top_p <= 0.f || top_p > 1.f) {
                report_invalid_argument(argv[i], argv[i + 1]);
            }
            i++;
        } else if (streq(argv[i], "--temperature")) {
            require_value(i, argc, argv[i]);
            temperature = parse_float(argv[i], argv[i + 1]);
            if (temperature <= 0.f) {
                report_invalid_argument(argv[i], argv[i + 1]);
            }
            i++;
        } else if (streq(argv[i], "--repetition-penalty")) {
            require_value(i, argc, argv[i]);
            repetition_penalty = parse_float(argv[i], argv[i + 1]);
            if (repetition_penalty < 1.f) {
                report_invalid_argument(argv[i], argv[i + 1]);
            }
            i++;
        } else if (streq(argv[i], "--frequency-penalty")) {
            require_value(i, argc, argv[i]);
            frequency_penalty = parse_float(argv[i], argv[i + 1]);
            if (frequency_penalty < -2.f || frequency_penalty > 2.f) {
                report_invalid_argument(argv[i], argv[i + 1]);
            }
            i++;
        } else if (streq(argv[i], "--presence-penalty")) {
            require_value(i, argc, argv[i]);
            presence_penalty = parse_float(argv[i], argv[i + 1]);
            if (presence_penalty < -2.f || presence_penalty > 2.f) {
                report_invalid_argument(argv[i], argv[i + 1]);
            }
            i++;
        } else if (streq(argv[i], "--seed")) {
            require_value(i, argc, argv[i]);
            random_seed = parse_int(argv[i], argv[i + 1]);
            if (random_seed < 0) {
                report_invalid_argument(argv[i], argv[i + 1]);
            }
            i++;
        } else if (streq(argv[i], "--rope-scaling-factor")) {
            require_value(i, argc, argv[i]);
            rope_scaling_factor = parse_float(argv[i], argv[i + 1]);
            if (rope_scaling_factor < 1.f || rope_scaling_factor > 8.f) {
                report_invalid_argument(argv[i], argv[i + 1]);
            }
            i++;
        } else {
            report_and_exit(std::cerr, 1);
        }
    }

    // default chat mode
    if (run_mode == mode::unknown) {
        run_mode = mode::chat;
    }

    if (show_debug_info) {
        std::cout << "[Debug] cli: top_k = " << top_k << std::endl;
        std::cout << "[Debug] cli: top_p = " << top_p << std::endl;
        std::cout << "[Debug] cli: temperature = " << temperature << std::endl;
        std::cout << "[Debug] cli: repetition_penalty = " << repetition_penalty << std::endl;
        std::cout << "[Debug] cli: frequency_penalty = " << frequency_penalty << std::endl;
        std::cout << "[Debug] cli: presence_penalty = " << presence_penalty << std::endl;
        if (random_seed == -1) {
            std::cout << "[Debug] cli: random_seed = std::random_device" << std::endl;
        } else {
            std::cout << "[Debug] cli: random_seed = " << random_seed << std::endl;
        }
        std::cout << "[Debug] cli: rope_scaling_factor = " << rope_scaling_factor << std::endl;
    }
}

}
