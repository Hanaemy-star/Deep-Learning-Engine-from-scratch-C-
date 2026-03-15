#pragma once

#include "tensor.hpp"

class Optimizer {
private:
    std::vector<std::shared_ptr<Tensor>> params;
    double lr;
public:
    Optimizer(std::vector<std::shared_ptr<Tensor>> params, double lr);

    void step();

    void zero_grad();
};
