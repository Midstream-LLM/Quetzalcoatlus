#include "util/perf_info.hpp"

namespace quetzal::util {

void perf_info::dump(std::ostream& os) const {
    os << "================================\n";
    os << "Performance (indices length: " << indices_length << "):\n";
    os << " - Transformer Block:\n";
    for (std::size_t i = 0; i < transformer_perf.size(); ++i) {
        const auto& tpi = transformer_perf[i];
        os << "  - block." << i << ": " << tpi.total_time << " ms ";
        os << "(" << tpi.total_time * 100.f / total_perf << "%)\n";
        os << "   - attn: " << tpi.attn_time.total_time.count() << " μs\n";
        os << "    - QKV     : " << tpi.attn_time.QKV_time.count() << " μs\n";
        os << "    - KV_cache: " << tpi.attn_time.KV_cache_time.count() << " ns\n";
        os << "    - QK_score: " << tpi.attn_time.QK_score_time.count() << " μs\n";
        os << "    - merge   : " << tpi.attn_time.merge_time.count() << " μs\n";
        os << "   - ffn : " << tpi.ffn_time.count() << " μs\n";
    }
    os << " - Layernorm   : " << layernorm_perf.count() << " μs ";
    os << "(" << layernorm_perf.count() / 1000.f / total_perf << "%)\n";
    os << " - Logits      : " << logits_calc_perf.count() << " μs ";
    os << "(" << logits_calc_perf.count() / 1000.f / total_perf << "%)\n";
    os << " - Choose token: " << token_choose_perf.count() << " μs ";
    os << "(" << token_choose_perf.count() / 1000.f / total_perf << "%)\n";
    os << " - Total       : " << total_perf << " ms\n";
    os << " - Alloc Count : " << alloc_count << "\n";
}

}
