#pragma once

#include <cmath>
#include <iostream>

namespace quetzal::test {
inline int test_count = 0;
inline int test_failed = 0;

inline void report_failure(const char* file, int line, const char* cond) {
    ++quetzal::test::test_failed;
    std::cerr << "[FAIL] " << file << ":" << line << ": " << cond << std::endl;
}

inline int report_final_result(const char* test_name) {
    std::cerr << (quetzal::test::test_failed == 0 ? "[PASS]" : "[FAIL]")
              << " " << test_name << ": " << quetzal::test::test_count << " tests, "
              << quetzal::test::test_failed << " failures" << std::endl;
    return quetzal::test::test_failed == 0 ? 0 : 1;
}

}

#define CHECK(cond) \
    do {                                                                           \
        ++quetzal::test::test_count;                                               \
        if (!(cond)) { quetzal::test::report_failure(__FILE__, __LINE__, #cond); } \
    } while(0)

#define CHECK_NEAR(a, b, eps) \
    do {                                                                     \
        ++quetzal::test::test_count;                                         \
        double a_ = (a), b_ = (b), eps_ = (eps);                             \
        if (!(std::fabs(a_ - b_) <= eps_)) {                                     \
            quetzal::test::report_failure(__FILE__, __LINE__, #a " == " #b); \
        }                                                                    \
    } while(0)

