#include "tensor/omp.hpp"
#include "tensor/assert.hpp"
#include "util/timestamp.hpp"

#include <cstdint>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>

void test_add(int size) {
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.f, 1.f);

    quetzal::util::timestamp ts;
    float* buff_a = new float[size];
    float* buff_b = new float[size];
    float* buff_c = new float[size];
    for (int i = 0; i < size; ++i) {
        buff_a[i] = dist(rng);
        buff_b[i] = dist(rng);
    }
    std::memset(buff_c, 0, sizeof(float) * size);

    std::int64_t total = INT64_MAX;
    for (int t = 0; t < 110; ++t) {
        ts.stamp();
        for (int i = 0; i < size; ++i) {
            buff_c[i] = buff_a[i] + buff_b[i];
        }
        auto dur = ts.elapsed_micro_seconds().count();
        if (t >= 10) {
            total = (std::min)(total, dur);
        }
    }
    std::cout << "add [" << size << "]: " << total << " μs\n";
    for (int i = 0; i < size; ++i) {
        QUETZAL_ASSERT(buff_c[i] == buff_a[i] + buff_b[i], "error");
    }

    delete[] buff_a;
    delete[] buff_b;
    delete[] buff_c;
}

void test_add_omp(int size) {
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.f, 1.f);

    quetzal::util::timestamp ts;
    float* buff_a = new float[size];
    float* buff_b = new float[size];
    float* buff_c = new float[size];
    for (int i = 0; i < size; ++i) {
        buff_a[i] = dist(rng);
        buff_b[i] = dist(rng);
    }
    std::memset(buff_c, 0, sizeof(float) * size);

    std::int64_t total = INT64_MAX;
    for (int t = 0; t < 110; ++t) {
        ts.stamp();
        OMP_FOR
        for (int i = 0; i < size; ++i) {
            buff_c[i] = buff_a[i] + buff_b[i];
        }
        auto dur = ts.elapsed_micro_seconds().count();
        if (t >= 10) {
            total = (std::min)(total, dur);
        }
    }
    std::cout << "add_omp [" << size << "]: " << total << " μs\n";
    for (int i = 0; i < size; ++i) {
        QUETZAL_ASSERT(buff_c[i] == buff_a[i] + buff_b[i], "error");
    }

    delete[] buff_a;
    delete[] buff_b;
    delete[] buff_c;
}

void test_div(int size) {
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.1f, 1.f);

    quetzal::util::timestamp ts;
    float* buff_a = new float[size];
    float* buff_b = new float[size];
    float* buff_c = new float[size];
    for (int i = 0; i < size; ++i) {
        buff_a[i] = dist(rng);
        buff_b[i] = dist(rng);
    }
    std::memset(buff_c, 0, sizeof(float) * size);

    std::int64_t total = INT64_MAX;
    for (int t = 0; t < 110; ++t) {
        ts.stamp();
        for (int i = 0; i < size; ++i) {
            buff_c[i] = buff_a[i] / buff_b[i];
        }
        auto dur = ts.elapsed_micro_seconds().count();
        if (t >= 10) {
            total = (std::min)(total, dur);
        }
    }
    std::cout << "div [" << size << "]: " << total << " μs\n";
    for (int i = 0; i < size; ++i) {
        QUETZAL_ASSERT(buff_c[i] == buff_a[i] / buff_b[i], "error");
    }

    delete[] buff_a;
    delete[] buff_b;
    delete[] buff_c;
}

void test_div_omp(int size) {
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.1f, 1.f);

    quetzal::util::timestamp ts;
    float* buff_a = new float[size];
    float* buff_b = new float[size];
    float* buff_c = new float[size];
    for (int i = 0; i < size; ++i) {
        buff_a[i] = dist(rng);
        buff_b[i] = dist(rng);
    }
    std::memset(buff_c, 0, sizeof(float) * size);

    std::int64_t total = INT64_MAX;
    for (int t = 0; t < 110; ++t) {
        ts.stamp();
        OMP_FOR
        for (int i = 0; i < size; ++i) {
            buff_c[i] = buff_a[i] / buff_b[i];
        }
        auto dur = ts.elapsed_micro_seconds().count();
        if (t >= 10) {
            total = (std::min)(total, dur);
        }
    }
    std::cout << "div_omp [" << size << "]: " << total << " μs\n";
    for (int i = 0; i < size; ++i) {
        QUETZAL_ASSERT(buff_c[i] == buff_a[i] / buff_b[i], "error");
    }

    delete[] buff_a;
    delete[] buff_b;
    delete[] buff_c;
}

int main() {
    std::cout << "=================================\n";
    for (int i = 4096; i < 64 * 1024 * 1024; i *= 2) {
        test_add(i);
    }
    std::cout << "=================================\n";
    for (int i = 4096; i < 64 * 1024 * 1024; i *= 2) {
        test_add_omp(i);
    }
    std::cout << "=================================\n";
    for (int i = 4096; i < 64 * 1024 * 1024; i *= 2) {
        test_div(i);
    }
    std::cout << "=================================\n";
    for (int i = 4096; i < 64 * 1024 * 1024; i *= 2) {
        test_div_omp(i);
    }
    std::cout << "=================================\n";
    return 0;
}