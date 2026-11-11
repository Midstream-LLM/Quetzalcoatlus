#include "model/riverstrike/attention.hpp"
#include "tensor/linalg.hpp"
#include "tensor/fusion.hpp"
#include "tensor/rope.hpp"
#include "util/timestamp.hpp"

#include <cmath>

namespace quetzal::gpt {

tensor::tensor<float>
multi_head_attention::forward_attn(const tensor::tensor<float>& x) const {
    QUETZAL_ASSERT(x.shape().size() == 2, "[mha] shape mismatch");
    // only calculate attention matrix, for visualization

    // (seq_len, d_model) @ (d_model, d_model) -> (seq_len, d_model)
    auto Q = tensor::matmul_2d<float>(x, Wq_pre_transposed);
    auto K = tensor::matmul_2d<float>(x, Wk_pre_transposed);

    const auto seq_len = x.shape()[0];
    const auto d_k = d_model_ / n_head_;

    // (seq_len, d_model) -> (seq_len, n_head, d_k) -> (n_head, seq_len, d_k)
    Q = Q.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();
    K = K.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();

    rope_.apply(Q);
    rope_.apply(K);

    // (n_head, seq_len, d_k) @ (n_head, d_k, seq_len) -> (n_head, seq_len, seq_len)
    auto scores = tensor::matmul_batch<float>(Q, K.transpose(1, 2));
    scores /= std::sqrt(d_k);
    return tensor::causal_mask_softmax_fusion<float>(scores);
}

tensor::tensor<float>
multi_head_attention::forward(const tensor::tensor<float>& x) const {
    QUETZAL_ASSERT(x.shape().size() == 2, "[mha] shape mismatch");
    // pytorch nn.Linear stores weights in (out_dim, in_dim)
    // but in forward progress, it still uses x @ (in_dim, out_dim)
    // so we need to transpose them here too

    // (seq_len, d_model) @ (d_model, d_model) -> (seq_len, d_model)
    auto Q = tensor::matmul_2d<float>(x, Wq_pre_transposed);
    auto K = tensor::matmul_2d<float>(x, Wk_pre_transposed);
    auto V = tensor::matmul_2d<float>(x, Wv_pre_transposed);

    const auto seq_len = x.shape()[0];
    const auto d_k = d_model_ / n_head_;

    // (seq_len, d_model) -> (seq_len, n_head, d_k) -> (n_head, seq_len, d_k)
    Q = Q.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();
    K = K.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();
    V = V.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();

    rope_.apply(Q);
    rope_.apply(K);

    // (n_head, seq_len, d_k) @ (n_head, d_k, seq_len) -> (n_head, seq_len, seq_len)
    auto scores = tensor::matmul_batch<float>(Q, K.transpose(1, 2).contiguous());
    scores /= std::sqrt(d_k);

    auto attn = tensor::causal_mask_softmax_fusion<float>(scores);
    // (n_head, seq_len, seq_len) @ (n_head, seq_len, d_k) -> (n_head, seq_len, d_k)
    auto out = tensor::matmul_batch<float>(attn, V);

    // (n_head, seq_len, d_k) -> (seq_len, n_head, d_k)
    out = out.transpose(0, 1).contiguous();
    // (seq_len, n_head, d_k) -> (seq_len, d_model)
    out = out.reshape({seq_len, d_model_});
    // (seq_len, d_model) @ (d_model, d_model) -> (seq_len, d_model)
    out = tensor::matmul_2d<float>(out, Wo_pre_transposed);
    return out;
}

tensor::tensor<float>
multi_head_attention::prefill(const tensor::tensor<float>& x) {
    QUETZAL_ASSERT(x.shape().size() == 2, "[mha] shape mismatch");
    // pytorch nn.Linear stores weights in (out_dim, in_dim)
    // but in forward progress, it still uses x @ (in_dim, out_dim)
    // so we need to transpose them here too

    // (seq_len, d_model) @ (d_model, d_model) -> (seq_len, d_model)
    auto Q = tensor::matmul_2d<float>(x, Wq_pre_transposed);
    auto K = tensor::matmul_2d<float>(x, Wk_pre_transposed);
    auto V = tensor::matmul_2d<float>(x, Wv_pre_transposed);

    const auto seq_len = x.shape()[0];
    QUETZAL_ASSERT(seq_len <= max_seq_len_, "[mha] sequence length out of range");
    const auto d_k = d_model_ / n_head_;

    // (seq_len, d_model) -> (seq_len, n_head, d_k) -> (n_head, seq_len, d_k)
    Q = Q.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();
    K = K.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();
    V = V.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();

    rope_.apply(Q);
    rope_.apply(K);

    cache.prefill(K, V, seq_len);

    // (n_head, seq_len, d_k) @ (n_head, d_k, seq_len) -> (n_head, seq_len, seq_len)
    auto scores = tensor::matmul_batch<float>(Q, K.transpose(1, 2).contiguous());
    scores /= std::sqrt(d_k);

    auto attn = tensor::causal_mask_softmax_fusion<float>(scores);
    // (n_head, seq_len, seq_len) @ (n_head, seq_len, d_k) -> (n_head, seq_len, d_k)
    auto out = tensor::matmul_batch<float>(attn, V);

    // (n_head, seq_len, d_k) -> (seq_len, n_head, d_k)
    out = out.transpose(0, 1).contiguous();
    // (seq_len, n_head, d_k) -> (seq_len, d_model)
    out = out.reshape({seq_len, d_model_});
    // (seq_len, d_model) @ (d_model, d_model) -> (seq_len, d_model)
    out = tensor::matmul_2d<float>(out, Wo_pre_transposed);
    return out;
}

tensor::tensor<float>
multi_head_attention::decode(const tensor::tensor<float>& x,
                             util::attn_perf_info* api) {
    QUETZAL_ASSERT(x.shape().size() == 2, "[mha] shape mismatch");
    QUETZAL_ASSERT(x.shape()[0] == 1, "[mha] shape mismatch");
    
    util::timestamp total_ts;
    util::timestamp ts;

    total_ts.stamp();

    // (1, d_model) @ (d_model, d_model) -> (1, d_model)
    ts.stamp();
    auto Q = tensor::matmul_2d<float>(x, Wq_pre_transposed);
    auto K = tensor::matmul_2d<float>(x, Wk_pre_transposed);
    auto V = tensor::matmul_2d<float>(x, Wv_pre_transposed);

    const auto seq_len = x.shape()[0];
    const auto d_k = d_model_ / n_head_;

    // (1, d_model) -> (1, n_head, d_k) -> (n_head, 1, d_k)
    Q = Q.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();
    K = K.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();
    V = V.reshape({seq_len, n_head_, d_k}).transpose(0, 1).contiguous();

    rope_.apply(Q, cache.get_cache_len());
    rope_.apply(K, cache.get_cache_len());
    if (api) {
        api->QKV_time = ts.elapsed_micro_seconds();
    }

    ts.stamp();
    cache.update(K, V);
    K = cache.get_k_cache();
    V = cache.get_v_cache();
    if (api) {
        api->KV_cache_time = ts.elapsed_nano_seconds();
    }

    // (n_head, seq_len, d_k) @ (n_head, d_k, seq_len) -> (n_head, seq_len, seq_len)
    ts.stamp();
    auto scores = tensor::matmul_batch<float>(Q, K.transpose(1, 2));
    scores /= std::sqrt(d_k);

    if (api) {
        api->QK_score_time = ts.elapsed_micro_seconds();
    }

    ts.stamp();
    auto attn = tensor::softmax<float>(scores);
    // (n_head, seq_len, seq_len) @ (n_head, seq_len, d_k) -> (n_head, seq_len, d_k)
    auto out = tensor::matmul_batch<float>(attn, V);

    // (n_head, seq_len, d_k) -> (seq_len, n_head, d_k)
    out = out.transpose(0, 1).contiguous();
    // (seq_len, n_head, d_k) -> (seq_len, d_model)
    out = out.reshape({seq_len, d_model_});
    // (seq_len, d_model) @ (d_model, d_model) -> (seq_len, d_model)
    out = tensor::matmul_2d<float>(out, Wo_pre_transposed);
    if (api) {
        api->merge_time = ts.elapsed_micro_seconds();
    }

    if (api) {
        api->total_time = total_ts.elapsed_micro_seconds();
    }
    return out;
}

}
