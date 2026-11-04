#pragma once

#include "tensor/tensor.hpp"

namespace quetzal::gpt {

class mha_kv_cache {
private:
    std::size_t d_model_;
    std::size_t n_head_;
    std::size_t max_seq_len_;
    tensor::tensor<float> K_cache;
    tensor::tensor<float> V_cache;
    std::size_t cache_len;

public:
    mha_kv_cache(const std::size_t d_model,
                 const std::size_t n_head,
                 const std::size_t max_seq_len) :
        d_model_(d_model), n_head_(n_head), max_seq_len_(max_seq_len),
        K_cache({n_head, max_seq_len, d_model / n_head}),
        V_cache({n_head, max_seq_len, d_model / n_head}), cache_len(0) {}
    void reset() {
        // if need to read for visualization, memset should be used here
        // otherwise the data is dirty
        // but it does not matter for prefill/decode
        cache_len = 0;
    }
    void prefill(const tensor::tensor<float>& K,
                 const tensor::tensor<float>& V,
                 const std::size_t seq_len);
    void update(const tensor::tensor<float>& K,
                const tensor::tensor<float>& V);
    tensor::tensor<float> get_k_cache() const {
        return K_cache.narrow(1, cache_len);
    }
    tensor::tensor<float> get_v_cache() const {
        return V_cache.narrow(1, cache_len);
    }
    std::size_t get_cache_len() const { return cache_len; }
};

}
