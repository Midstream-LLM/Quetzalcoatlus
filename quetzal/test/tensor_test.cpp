#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>

#include "tensor/tensor.hpp"
#include "tensor/linalg.hpp"
#include "tensor/fusion.hpp"
#include "tensor/init.hpp"
#include "tensor/rope.hpp"
#include "util/testing.hpp"

bool test_contiguous() {
    std::cout << "================== contiguous ==================" << std::endl;
    quetzal::tensor::tensor<float> a({2, 3, 4});
    quetzal::tensor::debug_init(a);
    if (a.transpose(0, 1).is_contiguous()) {
        std::cout << "[contiguous] FAIL [a.transpose(0, 1).is_contiguous()]" << std::endl;
        return false;
    }
    std::cout << "[contiguous] PASS [a.transpose(0, 1).is_contiguous()]" << std::endl;
    
    if (!a.transpose(0, 1).transpose(0, 1).is_contiguous()) {
        std::cout << "[contiguous] FAIL [a.transpose(0, 1).transpose(0, 1).is_contiguous()]" << std::endl;
        return false;
    }
    std::cout << "[contiguous] PASS [a.transpose(0, 1).transpose(0, 1).is_contiguous()]" << std::endl;

    a.transpose(0, 1).dump_info(std::cout);
    a.transpose(0, 1).dump(std::cout);
    a.transpose(0, 1).contiguous().dump_info(std::cout);
    a.transpose(0, 1).contiguous().dump(std::cout);

    a.transpose(0, 2).dump_info(std::cout);
    a.transpose(0, 2).dump(std::cout);
    a.transpose(0, 2).contiguous().dump_info(std::cout);
    a.transpose(0, 2).contiguous().dump(std::cout);

    a.transpose(1, 2).dump_info(std::cout);
    a.transpose(1, 2).dump(std::cout);
    a.transpose(1, 2).contiguous().dump_info(std::cout);
    a.transpose(1, 2).contiguous().dump(std::cout);
    return true;
}

bool test_dump() {
    std::cout << "==================== dump =====================" << std::endl;
    std::vector<std::size_t> shape = {2};
    quetzal::tensor::tensor<float> a(shape);
    quetzal::tensor::debug_init(a);

    std::stringstream ss1;
    a.dump(ss1);

    std::stringstream ss2;
    ss2 << "[" << a[0] << ", " << a[1] << "]" << std::endl;

    if (ss1.str() != ss2.str()) {
        std::cout << "[dump] FAIL [ss1.str() != ss2.str()]" << std::endl;
        return false;
    }
    std::cout << "[dump] PASS [ss1.str() == ss2.str()]" << std::endl;
    return true;
}

void test_transpose() {
    std::cout << "================== transpose ==================" << std::endl;
    quetzal::tensor::tensor<float> a({2, 3, 4});
    quetzal::tensor::debug_init(a);
    try {
        a.dump_info(std::cout);
        a.dump(std::cout);
        a.transpose(0, 1).dump_info(std::cout);
        a.transpose(0, 1).dump(std::cout);
    } catch (std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    quetzal::tensor::tensor<float> b({2, 3, 4});
    quetzal::tensor::debug_init(b);
    try {
        b.dump_info(std::cout);
        b.dump(std::cout);
        b.transpose(0, 2).dump_info(std::cout);
        b.transpose(0, 2).dump(std::cout);
    } catch (std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    quetzal::tensor::tensor<float> c({2, 3, 4});
    quetzal::tensor::debug_init(c);
    try {
        c.dump_info(std::cout);
        c.dump(std::cout);
        c.transpose(1, 2).dump_info(std::cout);
        c.transpose(1, 2).dump(std::cout);
    } catch (std::exception &e) {
        std::cout << e.what() << std::endl;
    }
}

void test_add() {
    std::cout << "===================== add =====================" << std::endl;
    quetzal::tensor::tensor<float> a({2, 3, 4});
    quetzal::tensor::debug_init(a, 10.0f);
    quetzal::tensor::tensor<float> b({2, 3, 4});
    quetzal::tensor::debug_init(b, 10.0f);
    quetzal::tensor::tensor<float> c = a + b;
    c.dump_info(std::cout);
    c.dump(std::cout);
}

void test_mul() {
    std::cout << "===================== mul =====================" << std::endl;
    quetzal::tensor::tensor<float> a({2, 3, 4});
    quetzal::tensor::debug_init(a, 10.0f);
    quetzal::tensor::tensor<float> b({2, 3, 4});
    quetzal::tensor::debug_init(b, 10.0f);
    quetzal::tensor::tensor<float> c = a * b;
    c.dump_info(std::cout);
    c.dump(std::cout);

    quetzal::tensor::tensor<float> d = c * 100.0f;
    d.dump_info(std::cout);
    d.dump(std::cout);
}

void test_silu() {
    std::cout << "===================== silu ====================" << std::endl;
    quetzal::tensor::tensor<float> a({2, 3, 4});
    quetzal::tensor::debug_init(a, 10.0f);
    quetzal::tensor::tensor<float> b = quetzal::tensor::silu(a);
    b.dump_info(std::cout);
    b.dump(std::cout);
}

bool test_silu_mul_fusion() {
    std::cout << "============== silu_mul_fusion ==============" << std::endl;
    const std::vector<std::vector<std::size_t>> shapes = {
        {16},
        {4, 64},
        {2, 3, 8},
    };
    for (const auto& shape : shapes) {
        quetzal::tensor::tensor<float> a(shape);
        quetzal::tensor::tensor<float> b(shape);
        quetzal::tensor::debug_init(a, 10.0f);
        quetzal::tensor::debug_init(b, 7.0f);

        quetzal::tensor::tensor<float> fused = quetzal::tensor::silu_mul_fusion(a, b);
        quetzal::tensor::tensor<float> ref   = quetzal::tensor::silu(a) * b;

        float max_diff = 0.0f;
        std::size_t arg = 0;
        for (std::size_t i = 0; i < fused.total_size(); ++i) {
            float diff = std::fabs(fused.data()[i] - ref.data()[i]);
            if (diff > max_diff) {
                max_diff = diff;
                arg = i;
            }
        }

        std::cout << "[silu_mul_fusion shape=";
        for (std::size_t d = 0; d < shape.size(); ++d) {
            std::cout << (d == 0 ? "" : "x") << shape[d];
        }
        std::cout << "] max|diff| = " << max_diff << " @ " << arg << std::endl;

        if (max_diff != 0.0f) {
            std::cout << "[silu_mul_fusion] FAIL" << std::endl;
            return false;
        }
    }
    std::cout << "[silu_mul_fusion] PASS (bit-identical to silu(a) * b)" << std::endl;
    return true;
}

void test_sigmoid() {
    std::cout << "=================== sigmoid ===================" << std::endl;
    quetzal::tensor::tensor<float> a({2, 3, 4});
    quetzal::tensor::debug_init(a, 10.0f);
    quetzal::tensor::tensor<float> b = quetzal::tensor::sigmoid(a);
    b.dump_info(std::cout);
    b.dump(std::cout);
}

void test_softmax() {
    std::cout << "=================== softmax ===================" << std::endl;
    quetzal::tensor::tensor<float> a({2, 3, 40});
    quetzal::tensor::debug_init(a, 240.0f);
    a.dump_info(std::cout);
    a.dump(std::cout);
    quetzal::tensor::tensor<float> b = quetzal::tensor::softmax(a);
    b.dump_info(std::cout);
    b.dump(std::cout);
}

void test_layer_norm() {
    std::cout << "================== layernorm ==================" << std::endl;
    quetzal::tensor::tensor<float> a({2, 3, 40});
    quetzal::tensor::debug_init(a, 240.0f);
    quetzal::tensor::tensor<float> weights({40});
    quetzal::tensor::debug_init(weights, 80.0f);
    quetzal::tensor::tensor<float> bias({40});
    quetzal::tensor::debug_init(bias, 80.0f);
    quetzal::tensor::tensor<float> b = quetzal::tensor::layernorm(a, weights, bias, 1e-5f);
    b.dump_info(std::cout);
    b.dump(std::cout);
}

void test_2d_matmul() {
    std::cout << "=================== 2d matmul ==================" << std::endl;
    quetzal::tensor::tensor<float> a({2, 3});
    quetzal::tensor::debug_init(a);
    quetzal::tensor::tensor<float> b({3, 4});
    quetzal::tensor::debug_init(b);
    quetzal::tensor::tensor<float> c = quetzal::tensor::matmul_2d(a, b);

    a.dump(std::cout);
    b.dump(std::cout);
    c.dump_info(std::cout);
    c.dump(std::cout);

    quetzal::tensor::tensor<float> d = b.transpose(0, 1).contiguous();
    quetzal::tensor::tensor<float> e = quetzal::tensor::matmul_2d(a, d.transpose(0, 1));

    e.dump_info(std::cout);
    e.dump(std::cout);
}

void test_2d_matmul_perf() {
    quetzal::tensor::tensor<float> a({352, 3520 * 4});
    quetzal::tensor::debug_init(a);
    quetzal::tensor::tensor<float> b({3520 * 4, 352});
    quetzal::tensor::debug_init(b);

    std::cout << "=============== 2d matmul (perf) ==============" << std::endl;
    std::chrono::high_resolution_clock::time_point t1, t2;
    t1 = std::chrono::high_resolution_clock::now();
    quetzal::tensor::tensor<float> c = quetzal::tensor::matmul_2d(a, b);
    t2 = std::chrono::high_resolution_clock::now();

    std::cout << "matmul_2d time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
              << " ms" << std::endl;

    c.dump_info(std::cout);
    c.dump(std::cout);
}

void test_matmul_batch() {
    std::cout << "================= matmul batch ================" << std::endl;
    quetzal::tensor::tensor<float> a({2, 3, 4});
    quetzal::tensor::tensor<float> b({2, 4, 2});
    quetzal::tensor::debug_init(a);
    quetzal::tensor::debug_init(b);

    quetzal::tensor::tensor<float> c = quetzal::tensor::matmul_batch(a, b);
    a.dump(std::cout);
    b.dump(std::cout);
    c.dump_info(std::cout);
    c.dump(std::cout);
}

void test_causal_mask() {
    std::cout << "=================== casual mask ==================" << std::endl;
    quetzal::tensor::tensor<float> a({2, 5, 5});
    quetzal::tensor::debug_init(a, 50.0f);
    quetzal::tensor::apply_causal_mask(a);

    a.dump_info(std::cout);
    a.dump(std::cout);
}

void test_reshape() {
    std::cout << "==================== reshape ==================" << std::endl;
    quetzal::tensor::tensor<float> a({2, 3, 4});
    quetzal::tensor::debug_init(a);
    quetzal::tensor::tensor<float> b = a.reshape({2, 12});

    a.dump_info(std::cout);
    a.dump(std::cout);
    b.dump_info(std::cout);
    b.dump(std::cout);

    try {
        auto tmp = a.transpose(1, 2).reshape({2, 12});
    } catch (std::exception &e) {
        std::cout << "expected exception:\n";
        std::cout << "  - " << e.what() << std::endl;
    }
}

bool test_embedding_gather() {
    std::cout << "=============== embedding gather ==============" << std::endl;
    quetzal::tensor::tensor<float> embedding({8, 352});
    quetzal::tensor::debug_init(embedding);
    std::vector<std::uint32_t> indices({0, 2, 4, 6});

    quetzal::tensor::tensor<float> got = quetzal::tensor::embedding_gather(embedding, indices);
    std::vector<float> expected;
    for (auto i : indices) {
        for (std::size_t j = 0; j < 352; ++j) {
            expected.push_back(i * 352.0f + j);
        }
    }

    bool ok = true;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (got.data()[i] != expected[i]) {
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "[embedding_gather] PASS" << std::endl;
    } else {
        std::cout << "[embedding_gather] FAIL" << std::endl;
    }
    return ok;
}

void test_rope() {
    std::cout << "===================== rope ====================" << std::endl;
    quetzal::tensor::tensor<float> a({2, 64, 44});
    quetzal::tensor::debug_init(a);
    
    quetzal::tensor::rope<float> rope(64, 44);
    rope.apply(a);

    a.dump_info(std::cout);
    a.dump(std::cout);
}

bool test_topk() {
    std::cout << "===================== topk ====================" << std::endl;
    quetzal::tensor::tensor<float> a({6});
    quetzal::tensor::debug_init(a);

    auto value = quetzal::tensor::topk(a, 5);
    if (value == 1.0f) {
        std::cout << "[topk] PASS" << std::endl;
    } else {
        std::cout << "[topk] FAIL" << std::endl;
    }
    return value == 1.0f;
}

bool test_multinomial() {
    std::cout << "================== multinomial ================" << std::endl;
    std::mt19937_64 rng(114514);

    quetzal::tensor::tensor<float> a({6});
    quetzal::tensor::zero(a);
    a.data()[0] = 1.0f;
    for (int i = 0; i < 1000; ++i) {
        if (quetzal::tensor::multinomial(a, rng) != 0) {
            std::cout << "[multinomial] FAIL [expect 0]" << std::endl;
            return false;
        }
    }

    quetzal::tensor::zero(a);
    a.data()[2] = 1.0f;
    for (int i = 0; i < 1000; ++i) {
        if (quetzal::tensor::multinomial(a, rng) != 2) {
            std::cout << "[multinomial] FAIL [expect 2]" << std::endl;
            return false;
        }
    }

    std::cout << "[multinomial] PASS" << std::endl;
    return true;
}

int main() {
    CHECK(test_dump());
    test_transpose();
    CHECK(test_contiguous());

    test_add();
    test_mul();
    test_silu();
    CHECK(test_silu_mul_fusion());
    test_sigmoid();
    test_softmax();
    test_layer_norm();
    test_2d_matmul();
    test_2d_matmul_perf();

    test_causal_mask();
    test_reshape();
    CHECK(test_embedding_gather());
    test_rope();

    test_matmul_batch();

    CHECK(test_topk());
    CHECK(test_multinomial());
    return quetzal::test::report_final_result("tensor_test");
}