#pragma once

#include <iostream>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <stdexcept>
#include <random>
#include <type_traits>
#include <vector>
#include <memory>

#include "assert.hpp"
#include "omp.hpp"

namespace quetzal::tensor {

template<typename T>
class tensor;

template<typename T>
class view {
private:
    friend class tensor<T>;
    const std::vector<std::size_t>& shape_;
    const std::vector<std::size_t>& strides_;
    const T* data_;
    std::size_t dim_;
    std::size_t offset_;

private:
    view(const std::vector<std::size_t>& shape,
         const std::vector<std::size_t>& strides,
         const T* data,
         std::size_t dim,
         std::size_t offset):
        shape_(shape), strides_(strides),
        data_(data), dim_(dim), offset_(offset) {}

    static void dump_indent(std::ostream& out, size_t indent) {
        for (std::size_t i = 0; i < indent; i++) {
            out << " ";
        }
    }

public:

    view operator[](std::size_t i) const {
        QUETZAL_ASSERT(dim_ < shape_.size(), "[view] dim out of range");
        QUETZAL_ASSERT(i < shape_[dim_], "[view] index out of range");

        std::size_t offset = offset_ + strides_[dim_] * i;
        return view<T>(shape_, strides_, data_, dim_ + 1, offset);
    }

    operator const T&() const {
        QUETZAL_ASSERT(dim_ == shape_.size(), "[view] dim out of range");
        return data_[offset_];
    }

    void dump(std::ostream& out, size_t indent = 0) {
        if (dim_ == shape_.size()) {
            out << data_[offset_];
            return;
        }

        const bool not_last_dim = (shape_.size() > 1 && dim_ < shape_.size() - 1);
        dump_indent(out, indent);
        out << "[";
        if (not_last_dim) {
            out << "\n";
        }

        bool dumped = false;
        for (std::size_t i = 0; i < shape_[dim_]; i++) {
            if (i != shape_[dim_] - 1 && i > 8) {
                if (!dumped) {
                    if (not_last_dim) {
                        dump_indent(out, indent + 1);
                    }
                    out << "..." << (not_last_dim ? "\n" : " ");
                }
                dumped = true;
                continue;
            }
            (*this)[i].dump(out, indent + 1);
            if (i != shape_[dim_] - 1) {
                out << ", ";
            }
            if (not_last_dim) {
                out << "\n";
            }
        }
        if (not_last_dim) {
            dump_indent(out, indent);
        }
        out << "]";
    }
};

using alloc_func = void* (*)(std::size_t);
using free_func = void (*)(void*);

void clear_alloc_count();
std::size_t get_alloc_count();
void *default_allocator(std::size_t size);
void default_deallocator(void *ptr);

template<typename T>
class tensor {
    static_assert(std::is_floating_point<T>::value, "T must be floating point type");
private:
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::shared_ptr<void> data_;

private:
    void compute_strides() {
        strides_.resize(shape_.size());
        std::size_t stride = 1;
        for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; i--) {
            strides_[i] = stride;
            stride *= shape_[i];
        }
    }

public:
    std::size_t total_size() const {
        std::size_t n = 1;
        for (auto& i : shape_) {
            n *= i;
        }
        return n;
    }

public:
    tensor(const std::vector<std::size_t>& shape,
           alloc_func allocator = default_allocator,
           free_func deallocator = default_deallocator): shape_(shape) {
        std::size_t s = total_size() * sizeof(T);
        data_ = std::shared_ptr<void>(allocator(s), deallocator);
        compute_strides();
    }

    tensor(const std::vector<std::size_t>& shape,
           const std::vector<T>& data,
           alloc_func allocator = default_allocator,
           free_func deallocator = default_deallocator): shape_(shape) {
        std::size_t s = total_size() * sizeof(T);
        data_ = std::shared_ptr<void>(allocator(s), deallocator);
        std::memcpy(data_.get(), data.data(), s);
        compute_strides();
    }

    tensor(std::initializer_list<std::size_t> shape,
           alloc_func allocator = default_allocator,
           free_func deallocator = default_deallocator): shape_(shape) {
        std::size_t n = total_size() * sizeof(T);
        data_ = std::shared_ptr<void>(allocator(n), deallocator);
        compute_strides();
    }

    tensor(const tensor& other) = default;

    static bool shape_equal(const tensor<T>& a, const tensor<T>& b) {
        return a.shape() == b.shape();
    }

    view<T> operator[](std::size_t i) const & {
        QUETZAL_ASSERT(!shape_.empty(), "[tensor] shape is empty");
        QUETZAL_ASSERT(i < shape_[0], "[tensor] index out of range");

        std::size_t offset = strides_[0] * i;
        return view<T>(shape_, strides_, data(), 1, offset);
    }

    view<T> operator[](std::size_t) && = delete;
    view<T> operator[](std::size_t i) const && = delete;

    T* data() { return static_cast<T*>(data_.get()); }
    const T* data() const { return static_cast<const T*>(data_.get()); }
    std::size_t ndim() const { return shape_.size(); }
    const std::vector<std::size_t>& shape() const { return shape_; }
    std::size_t shape(std::size_t i) const { return shape_.at(i); }
    const std::vector<std::size_t>& strides() const { return strides_; }
    std::size_t strides(std::size_t i) const { return strides_.at(i); }

public:
    tensor<T> transpose(std::size_t i, std::size_t j) const {
        if (i == j) {
            return *this;
        }

        if (i >= shape_.size() || j >= shape_.size()) {
            throw std::out_of_range("[transpose] dimension out of range");
        }

        tensor<T> ret = *this;
        std::swap(ret.shape_[i], ret.shape_[j]);
        std::swap(ret.strides_[i], ret.strides_[j]);
        return ret;
    }

    tensor<T> reshape(const std::vector<std::size_t>& shape) const {
        QUETZAL_ASSERT(is_contiguous(), "[reshape] tensor must be contiguous");
        std::size_t n = 1;
        for (auto& i : shape) {
            n *= i;
        }
        if (n != total_size()) {
            throw std::invalid_argument(
                "[reshape] total size of new array must be unchanged: " +
                std::to_string(total_size()) +
                " -> " +
                std::to_string(n));
        }

        tensor<T> ret = *this;
        ret.shape_ = shape;
        ret.compute_strides();
        return ret;
    }

public:
    bool is_contiguous() const {
        std::size_t stride = 1;
        for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; i--) {
            if (strides_[i] != stride) {
                return false;
            }
            stride *= shape_[i];
        }
        return true;
    }

    tensor<T> contiguous() const {
        if (is_contiguous()) {
            return *this;
        }

        tensor<T> ret = tensor<T>(shape_);
        const std::size_t total = total_size();

        std::vector<std::size_t> idx;
        idx.resize(shape_.size());

        for (std::size_t i = 0; i < total; i++) {
            std::size_t rem = i;
            for (std::size_t j = 0; j < shape_.size(); j++) {
                idx[j] = rem / ret.strides_[j];
                rem %= ret.strides_[j];
            }
            std::size_t offset = 0;
            for (std::size_t j = 0; j < shape_.size(); j++) {
                offset += idx[j] * strides_[j];
            }
            ret.data()[i] = data()[offset];
        }

        return ret;
    }

    tensor<T> narrow(std::size_t dim, std::size_t len) const {
        QUETZAL_ASSERT(dim < shape_.size(), "[narrow] dim out of range");
        QUETZAL_ASSERT(len <= shape_[dim], "[narrow] range out of bounds");

        tensor<T> ret = *this;
        ret.shape_[dim] = len;
        return ret;
    }

    tensor<T> operator+(const tensor<T>& b) const {
        QUETZAL_ASSERT(shape_equal(*this, b), "[add] shape mismatch");
        QUETZAL_ASSERT(this->is_contiguous() && b.is_contiguous(), "[add] tensors must be contiguous");

        std::size_t n = this->total_size();
        tensor<T> c(this->shape());

        for (std::size_t i = 0; i < n; ++i) {
            c.data()[i] = this->data()[i] + b.data()[i];
        }

        return c;
    }

    tensor<T>& operator+=(const tensor<T>& b) {
        QUETZAL_ASSERT(shape_equal(*this, b), "[add] shape mismatch");
        QUETZAL_ASSERT(this->is_contiguous() && b.is_contiguous(), "[add] tensors must be contiguous");
        QUETZAL_ASSERT(this->data() != b.data(), "[add] trying to add the same tensor");

        std::size_t n = this->total_size();

        for (std::size_t i = 0; i < n; ++i) {
            this->data()[i] += b.data()[i];
        }

        return *this;
    }

    tensor<T> operator-(const tensor<T>& b) const {
        QUETZAL_ASSERT(shape_equal(*this, b), "[sub] shape mismatch");
        QUETZAL_ASSERT(this->is_contiguous() && b.is_contiguous(), "[sub] tensors must be contiguous");

        std::size_t n = this->total_size();
        tensor<T> c(this->shape());

        for (std::size_t i = 0; i < n; ++i) {
            c.data()[i] = this->data()[i] - b.data()[i];
        }

        return c;
    }

    tensor<T> operator*(const tensor<T>& b) const {
        QUETZAL_ASSERT(shape_equal(*this, b), "[mul] shape mismatch");
        QUETZAL_ASSERT(this->is_contiguous() && b.is_contiguous(), "[mul] tensors must be contiguous");

        std::size_t n = this->total_size();
        tensor<T> c(this->shape());

        for (std::size_t i = 0; i < n; ++i) {
            c.data()[i] = this->data()[i] * b.data()[i];
        }

        return c;
    }

    tensor<T> operator*(T b) const {
        QUETZAL_ASSERT(this->is_contiguous(), "[mul] tensor must be contiguous");

        std::size_t n = this->total_size();
        tensor<T> c(this->shape());

        for (std::size_t i = 0; i < n; ++i) {
            c.data()[i] = this->data()[i] * b;
        }

        return c;
    }

    tensor<T> operator/(const tensor<T>& b) const {
        QUETZAL_ASSERT(shape_equal(*this, b), "[div] shape mismatch");
        QUETZAL_ASSERT(this->is_contiguous() && b.is_contiguous(), "[div] tensors must be contiguous");

        std::size_t n = this->total_size();
        tensor<T> c(this->shape());

        for (std::size_t i = 0; i < n; ++i) {
            c.data()[i] = this->data()[i] / b.data()[i];
        }

        return c;
    }

    tensor<T> operator/(T b) const {
        QUETZAL_ASSERT(this->is_contiguous(), "[div] tensor must be contiguous");
        QUETZAL_ASSERT(b != T(0), "[div] divisor is zero");

        std::size_t n = this->total_size();
        tensor<T> c(this->shape());

        for (std::size_t i = 0; i < n; ++i) {
            c.data()[i] = this->data()[i] / b;
        }

        return c;
    }

    tensor<T>& operator/=(T b) {
        QUETZAL_ASSERT(this->is_contiguous(), "[div] tensor must be contiguous");
        QUETZAL_ASSERT(b != T(0), "[div] divisor is zero");

        std::size_t n = this->total_size();

        for (std::size_t i = 0; i < n; ++i) {
            this->data()[i] /= b;
        }

        return *this;
    }

public:
    void dump_info(std::ostream& out) const {
        out << "shape: [ ";
        for (auto& i : shape_) {
            out << i << " ";
        }
        out << "], ";

        out << "strides: [ ";
        for (auto& i : strides_) {
            out << i << " ";
        }
        out << "]" << std::endl;
    }

    void dump(std::ostream& out, size_t indent = 0) const {
        if (shape_.size() == 1) {
            out << "[";
            bool dumped = false;
            for (std::size_t i = 0; i < shape_[0]; i++) {
                if (i != shape_[0] - 1 && i > 8) {
                    if (!dumped) {
                        for (std::size_t j = 0; j < indent + 1; j++) {
                            out << " ";
                        }
                        out << "... ";
                    }
                    dumped = true;
                    continue;
                }
                out << data()[i];
                if (i != shape_[0] - 1) {
                    out << ", ";
                }
            }
            out << "]" << std::endl;
            return;
        }

        for (std::size_t i = 0; i < indent; i++) {
            out << " ";
        }
        out << "[\n";
        bool dumped = false;
        for (std::size_t i = 0; i < shape_[0]; i++) {
            if (i != shape_[0] - 1 && i > 8) {
                if (!dumped) {
                    for (std::size_t j = 0; j < indent + 1; j++) {
                        out << " ";
                    }
                    out << "...\n";
                }
                dumped = true;
                continue;
            }
            (*this)[i].dump(out, indent + 1);
            if (i != shape_[0] - 1) {
                out << ",";
            }
            out << "\n";
        }
        for (std::size_t i = 0; i < indent; i++) {
            out << " ";
        }
        out << "]" << std::endl;
    }
};

}
