#include "model/riverstrike/transformer.hpp"
#include "tensor/linalg.hpp"
#include "tensor/fusion.hpp"
#include "util/timestamp.hpp"

namespace quetzal::gpt {

tensor::tensor<float> swiglu::forward(const tensor::tensor<float>& x) const {
    auto gate_input = tensor::matmul_2d<float>(x, gate_proj_pre_transposed);
    auto up = tensor::matmul_2d<float>(x, up_proj_pre_transposed);
    auto down_input = tensor::silu_mul_fusion(gate_input, up);
    auto down = tensor::matmul_2d<float>(down_input, down_proj_pre_transposed);
    return down;
}

tensor::tensor<float> transformer::forward(const tensor::tensor<float>& x) const {
    auto res = x;
    auto attn_out = attn_.forward(tensor::layernorm<float>(res, ln1_w_, ln1_b_));
    res += attn_out;
    auto ffn_out = ffn_.forward(tensor::layernorm<float>(res, ln2_w_, ln2_b_));
    res += ffn_out;
    return res;
}

tensor::tensor<float> transformer::prefill(const tensor::tensor<float>& x) {
    auto res = x;
    auto attn_out = attn_.prefill(tensor::layernorm<float>(res, ln1_w_, ln1_b_));
    res += attn_out;
    auto ffn_out = ffn_.forward(tensor::layernorm<float>(res, ln2_w_, ln2_b_));
    res += ffn_out;
    return res;
}

tensor::tensor<float> transformer::decode(const tensor::tensor<float>& x,
                                          util::transformer_perf_info* tpi) {
    util::timestamp total_ts;
    util::timestamp ts;

    total_ts.stamp();

    auto res = x;
    auto attn_out = attn_.decode(tensor::layernorm<float>(res, ln1_w_, ln1_b_),
                                 tpi ? &tpi->attn_time : nullptr);
    res += attn_out;

    ts.stamp();
    auto ffn_out = ffn_.forward(tensor::layernorm<float>(res, ln2_w_, ln2_b_));
    if (tpi) {
        tpi->ffn_time = ts.elapsed_micro_seconds();
    }

    res += ffn_out;
    if (tpi) {
        tpi->total_time = total_ts.elapsed_micro_seconds().count() / 1000.f;
    }

    return res;
}

}
