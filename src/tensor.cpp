#include "tensor.hpp"
#include <numeric>

Tensor::Tensor(std::vector<size_t> shape, double initial_value) : shape(shape) {
    size_t total_size = calculate_size(shape);
    data = std::vector<double>(total_size, initial_value);
}

size_t Tensor::calculate_size(const std::vector<size_t>& s) {
    if (s.empty()) return 0;
    size_t res = 1;
    for (auto dim : s) res *= dim;
    return res;
}

double& Tensor::operator()(const std::vector<size_t>& indices) {
    size_t flat_index = 0;
    size_t strides = 1;
    for (int i = indices.size() - 1; i >= 0; i--) {
        flat_index += indices[i] * strides;
        strides *= shape[i];
    }
    return data[flat_index];
}

void Tensor::reshape(std::vector<size_t> nshape) {
    if (calculate_size(nshape) != data.size()) {
        throw std::invalid_argument("New shape must have the same total number of elements");
    }
    shape = nshape;
}