#include "tensor/init.hpp"
#include "tensor/linalg.hpp"
#include "tensor/tensor.hpp"
#include "tensor/fusion.hpp"
#include "util/testing.hpp"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>

using namespace quetzal;

namespace {

float max_abs_diff(const tensor::tensor<float>& a,
                   const tensor::tensor<float>& b,
                   std::size_t& idx) {
    std::size_t n = a.total_size();
    float m = 0.0f;
    idx = 0;
    for (std::size_t i = 0; i < n; ++i) {
        float d = std::fabs(a.data()[i] - b.data()[i]);
        if (!(d <= m)) {
            m = d;
            idx = i;
        }
    }
    return m;
}

} // namespace

bool test_softmax_causal_fusion() {
    // (n_head, seq) shapes, incl. boundary seq=1
    const struct {
        std::size_t n_head;
        std::size_t seq;
    } shapes[] = {
        {1, 1}, {1, 4}, {2, 8}, {4, 64},
    };

    bool ok = true;

    for (const auto& s : shapes) {
        const std::size_t n_head = s.n_head;
        const std::size_t seq = s.seq;

        // monotonic increasing input: upper triangle (future) holds LARGER
        // values than the causal window, exactly the case that stresses the
        // max-scan bound
        tensor::tensor<float> scores({n_head, seq, seq});
        tensor::debug_init(scores, 1000.0f);

        // reference: apply_causal_mask + softmax.
        // NOTE tensor's copy ctor shares the data buffer (shared_ptr), so
        // deep-copy before the in-place mask.
        tensor::tensor<float> ref({n_head, seq, seq});
        std::memcpy(ref.data(), scores.data(), scores.total_size() * sizeof(float));
        tensor::apply_causal_mask(ref);
        ref = tensor::softmax(ref);

        // fused
        tensor::tensor<float> fused = tensor::causal_mask_softmax_fusion(scores);

        std::size_t idx = 0;
        float d = max_abs_diff(fused, ref, idx);
        std::cout << "[softmax_causal n_head=" << n_head << " seq=" << seq
                  << "] max|diff| = " << std::scientific << d << " @ " << idx
                  << std::endl;
        if (d != 0.0f) ok = false;
    }

    return ok;
}

int main() {
    CHECK(test_softmax_causal_fusion());
    return test::report_final_result("softmax_causal");
}
