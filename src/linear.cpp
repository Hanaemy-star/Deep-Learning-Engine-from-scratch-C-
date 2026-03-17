#include "linear.hpp"
#include <random>

Linear::Linear(size_t in_features, size_t out_features) {
    W = std::make_shared<Tensor>(std::vector<size_t>{in_features, out_features}, 0.0, true);
    B = std::make_shared<Tensor>(std::vector<size_t>{1, out_features}, 0.0, true);

    std::random_device rd;
    std::mt19937 gen(rd());
    double std_dev = std::sqrt(2.0 / (in_features + out_features));
    std::normal_distribution<double> dis(0.0, std_dev);

    auto& w_data = W->get_data();
    for (double& v : w_data) {
        v = dis(gen);
    }
}

std::shared_ptr<Tensor> Linear::forward(std::shared_ptr<Tensor> input) {
    return (input * W) + B;
}

std::vector<std::shared_ptr<Tensor> > Linear::parameters() {
    return {W, B};
}
