#include "model/riverstrike/attention.hpp"
#include "tensor/init.hpp"
#include "tensor/linalg.hpp"
#include "tensor/rope.hpp"
#include "util/testing.hpp"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>

using namespace quetzal;

namespace {

float max_abs_diff(const tensor::tensor<float>& a,
                   const float* b,
                   std::size_t n,
                   std::size_t& idx) {
    float m = 0.0f;
    idx = 0;
    for (std::size_t i = 0; i < n; ++i) {
        float d = std::fabs(a.data()[i] - b[i]);
        if (!(d <= m)) {
            m = d;
            idx = i;
        }
    }
    return m;
}

tensor::tensor<float> prefix(const tensor::tensor<float>& full, std::size_t rows) {
    std::size_t d = full.shape(1);
    tensor::tensor<float> t({rows, d});
    std::memcpy(t.data(), full.data(), rows * d * sizeof(float));
    return t;
}

tensor::tensor<float> row(const tensor::tensor<float>& full, std::size_t r) {
    std::size_t d = full.shape(1);
    tensor::tensor<float> t({1, d});
    std::memcpy(t.data(), full.data() + r * d, d * sizeof(float));
    return t;
}

} // namespace

bool test_kv_cache() {
    const std::size_t d_model = 64, n_head = 4, d_k = d_model / n_head;
    const std::size_t max_seq_len = 64;
    const std::size_t N = 12; // prefill tokens
    const std::size_t M = 8;  // decode tokens

    tensor::rope<float> rope(max_seq_len, d_k);

    tensor::tensor<float> Wq({d_model, d_model}), Wk({d_model, d_model}),
                          Wv({d_model, d_model}), Wo({d_model, d_model});
    // keep magnitudes small so attention logits stay well inside exp() range
    tensor::debug_init(Wq, 100000.0f);
    tensor::debug_init(Wk, 100000.0f);
    tensor::debug_init(Wv, 100000.0f);
    tensor::debug_init(Wo, 100000.0f);

    // reference instance (only const forward()) and cache instance
    gpt::multi_head_attention ref(d_model, n_head, max_seq_len, rope, Wq, Wk, Wv, Wo);
    gpt::multi_head_attention mha(d_model, n_head, max_seq_len, rope, Wq, Wk, Wv, Wo);

    // deterministic input, N + M tokens
    tensor::tensor<float> x_seq({N + M, d_model});
    tensor::debug_init(x_seq, 10000.0f);

    bool ok = true;

    // 1) prefill(N) must equal forward(x[0:N]) element-for-element
    {
        auto p = mha.prefill(prefix(x_seq, N));
        auto r = ref.forward(prefix(x_seq, N));
        std::size_t idx = 0;
        float d = max_abs_diff(p, r.data(), N * d_model, idx);
        std::cout << "[prefill vs forward]        max|diff| = " << std::scientific << d
                  << " @ " << idx << std::endl;
        if (d != 0.0f) ok = false;
    }

    // 2) each decode step must equal forward(x[0:N+1+k])'s last row
    for (std::size_t k = 0; k < M; ++k) {
        auto d = mha.decode(row(x_seq, N + k));
        auto r = ref.forward(prefix(x_seq, N + 1 + k));
        std::size_t idx = 0;
        float diff = max_abs_diff(d, r.data() + (N + k) * d_model, d_model, idx);
        std::cout << "[decode step " << k << " vs forward] max|diff| = " << std::scientific
                  << diff << " @ " << idx << std::endl;
        if (diff != 0.0f) ok = false;
    }

    return ok;
}

int main() {
    CHECK(test_kv_cache());
    return test::summary("kv_cache_verify");
}
