#pragma once

#include "tensor/tensor.hpp"

#include <fstream>
#include <string>
#include <vector>

namespace quetzal::util {

class ppm_writer {
private:
    std::ofstream out_;
    std::size_t width_;
    std::size_t height_;
    std::size_t total_bytes_;
    std::vector<tensor::tensor<float>> tensors;
    bool enable_debug_info_ = false;

public:
    void write_pixel(unsigned char r, unsigned char g, unsigned char b) {
        ++total_bytes_;
        out_ << r << g << b;
    }
    void write_signed(float value, float scale);

public:
    ppm_writer(const std::string& path,
               std::size_t width,
               std::size_t height,
               bool debug = false);
    ~ppm_writer();
    void write();
    void add(const tensor::tensor<float>& x);
    const std::vector<tensor::tensor<float>>& bands() const { return tensors; }
};

}
