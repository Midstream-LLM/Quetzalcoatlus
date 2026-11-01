#include "util/ppm.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace quetzal::util {

ppm_writer::ppm_writer(const std::string& path,
                       std::size_t width,
                       std::size_t height,
                       bool debug) :
    out_(path, std::ios::binary),
    width_(width), height_(height), total_bytes_(0),
    enable_debug_info_(debug) {
    if (!out_) {
        throw std::runtime_error("Could not open file " + path);
    }
    out_ << "P6\n" << width << " " << height << "\n255\n";
}

ppm_writer::~ppm_writer() {
    out_.close();
    if (enable_debug_info_) {
        std::cout << "[Debug] ppm_writer: write " << total_bytes_ << " bytes"
                  << " | " << width_ << "x" << height_
                  << " | actual height=" << total_bytes_ / width_ << "\n";
    }
}

void ppm_writer::write_signed(float value, float scale) {
    static const float mid[3] = { 26.f,  26.f,  26.f};
    static const float pos[3] = {230.f, 120.f,  50.f};
    static const float neg[3] = {76.f, 150.f, 235.f};

    const float t = std::clamp(value / scale, -1.f, 1.f);
    const float u = std::sqrt(std::abs(t));
    const float* end = t >= 0.f ? pos : neg;

    unsigned char rgb[3];
    for (std::size_t c = 0; c < 3; ++c) {
        rgb[c] = static_cast<unsigned char>(std::lround(mid[c] * (1.f - u) + end[c] * u));
    }
    write_pixel(rgb[0], rgb[1], rgb[2]);
}

void ppm_writer::write() {
    std::size_t band = 0;
    for (const auto& x : tensors) {
        for (std::size_t i = 0; i < width_; ++i) {
            write_pixel(0, 0, 0);
        }

        std::vector<float> mag(x.total_size());
        double sq = 0.0;
        for (std::size_t i = 0; i < x.total_size(); ++i) {
            mag[i] = std::abs(x.data()[i]);
            sq += static_cast<double>(x.data()[i]) * static_cast<double>(x.data()[i]);
        }
        const float rms = static_cast<float>(std::sqrt(sq / static_cast<double>(mag.size())));
        const float max_num = *std::max_element(mag.begin(), mag.end());
        const std::size_t k = static_cast<std::size_t>(0.995f * static_cast<float>(mag.size() - 1));
        std::nth_element(mag.begin(), mag.begin() + k, mag.end());
        const float q = mag[k] > 0.f ? mag[k] : 1.f;
        if (enable_debug_info_) {
            std::cout << "[Debug] ppm_writer: band " << band << " rms=" << rms
                      << " p99.5=" << q << " max=" << max_num << "\n";
        }
        ++band;

        for (std::size_t i = 0; i < x.total_size(); ++i) {
            write_signed(x.data()[i], q);
        }
    }
}

void ppm_writer::add(const tensor::tensor<float>& x) {
    tensors.push_back(x);
}

}
