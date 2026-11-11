#pragma once

#include <iostream>
#include <vector>
#include <chrono>

namespace quetzal::util {

struct attn_perf_info {
    std::chrono::microseconds QKV_time;
    std::chrono::nanoseconds  KV_cache_time;
    std::chrono::microseconds QK_score_time;
    std::chrono::microseconds merge_time;
    std::chrono::microseconds total_time;
};

struct transformer_perf_info {
    attn_perf_info attn_time;
    std::chrono::microseconds ffn_time;
    float total_time = 0.f;
};

struct perf_info {
    std::size_t indices_length = 0;
    std::vector<transformer_perf_info> transformer_perf;
    std::chrono::microseconds layernorm_perf;
    std::chrono::microseconds logits_calc_perf;
    std::chrono::microseconds token_choose_perf;
    float total_perf = 0.f;
    std::size_t alloc_count = 0;

    void dump(std::ostream&) const;
};

}
