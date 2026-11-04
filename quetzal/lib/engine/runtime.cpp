#include "engine/runtime.hpp"

namespace quetzal::runtime {

static std::optional<tensor::NTK_aware_scaling> make_rule(const util::cli& cli) {
    if (cli.get_rope_scaling_factor() <= 1.f) {
        return std::nullopt;
    }
    return tensor::NTK_aware_scaling(cli.get_rope_scaling_factor());
}

runtime::runtime(const util::cli& cli):
    qbr(cli.get_model_file_path(), cli.enable_debug_info()),
    tokenizer(qbr.get_bin_reader()),
    real_max_seq_len(qbr.get_model_config().max_seq_len *
                     cli.get_rope_scaling_factor()),
    rule(make_rule(cli)),
    model(qbr.get_weights_manager(),
          qbr.get_model_config(),
          rule ? &*rule : nullptr),
    penalty(cli.get_repetition_penalty(),
            cli.get_frequency_penalty(),
            cli.get_presence_penalty()) {
    qbr.release_memory();
}

}
