#include "optimizer.hpp"

Optimizer::Optimizer(std::vector<std::shared_ptr<Tensor> > params, double lr) : params(params), lr(lr){
}

void Optimizer::zero_grad() {
    for (auto p : params) {
        p->get_grad()->fill(0.0);
    }
}

void Optimizer::step() {
    for (auto& p : params) {
        if (p->get_grad()) {
            auto& grad =  p->get_grad()->get_data();
            auto& data = p->get_data();

            for (size_t i = 0; i < data.size(); i++) {
                data[i] -= lr * grad[i];
            }
        }
    }
}