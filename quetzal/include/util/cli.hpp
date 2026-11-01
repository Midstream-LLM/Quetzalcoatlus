#pragma once

#include <cstdlib>
#include <string>
#include <iostream>

namespace quetzal::util {

class cli {
public:
    enum class mode {
        unknown,
        chat,
        experimental,
        attn,
        perf
    };

private:
    std::string executable_name;
    std::string model_file_path;
    std::string system_prompt = "";
    mode run_mode = mode::unknown;
    float temperature = 0.8f;
    float repetition_penalty = 1.15f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    std::size_t top_k = 20;
    float top_p = 0.9f;
    bool disable_system_prompt = false;
    bool show_debug_info = false;
    int random_seed = -1;
    float rope_scaling_factor = 1.f;

private:
    void help(std::ostream& out) const;
    void print_version(std::ostream& out) const;

    [[noreturn]]
    void report_and_exit(std::ostream& out, int exit_code = 1) const {
        logo(out);
        help(out);
        std::exit(exit_code);
    }

    [[noreturn]]
    void report_lack_argument(const char* arg) const {
        std::cerr << "Error: Missing argument for option: \"" << arg << "\"\n";
        std::exit(1);
    }

    [[noreturn]]
    void report_invalid_argument(const char* arg, const char* value) const {
        std::cerr << "Error: Invalid argument for option: \"" << arg << "\": "
                  << "'" << value << "'\n";
        std::exit(1);
    }
    void require_value(int i, int argc, const char* arg) const {
        if (i + 1 >= argc) {
            report_lack_argument(arg);
        }
    }
    float parse_float(const char* arg, const char* value) const;
    int parse_int(const char* arg, const char* value) const;

public:
    cli(int argc, const char** argv);
    const char* version() const;
    void logo(std::ostream& os) const;
    const std::string& get_model_file_path() const {
        return model_file_path;
    }
    const std::string& get_system_prompt() const {
        return system_prompt;
    }
    bool is_chat_mode() const {
        return run_mode == mode::chat;
    }
    bool is_experimental_mode() const {
        return run_mode == mode::experimental;
    }
    bool is_attn_mode() const {
        return run_mode == mode::attn;
    }
    bool is_perf_mode() const {
        return run_mode == mode::perf;
    }
    const char* get_mode_name() const {
        switch (run_mode) {
            case mode::chat:
                return "chat";
            case mode::experimental:
                return "experimental";
            case mode::attn:
                return "attn";
            case mode::perf:
                return "perf";
            default:
                return "unknown";
        }
        return "unknown";
    }
    float get_temperature() const {
        return temperature;
    }
    float get_repetition_penalty() const {
        return repetition_penalty;
    }
    float get_frequency_penalty() const {
        return frequency_penalty;
    }
    float get_presence_penalty() const {
        return presence_penalty;
    }
    std::size_t get_top_k() const {
        return top_k;
    }
    float get_top_p() const {
        return top_p;
    }
    int get_random_seed() const {
        return random_seed;
    }
    float get_rope_scaling_factor() const {
        return rope_scaling_factor;
    }
    bool system_prompt_disabled() const {
        return disable_system_prompt;
    }
    bool enable_debug_info() const {
        return show_debug_info;
    }
};

}
