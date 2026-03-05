#include "tensor.hpp"

int main() {
    try {
        auto X = std::make_shared<Tensor>(std::vector<size_t>{1, 2}, 0.0, false);
        X->get_data() = {1.0, 2.0};

        auto W = std::make_shared<Tensor>(std::vector<size_t>{2, 2}, 0.0, true);
        W->get_data() = {2.0, 0.0, 0.0, 3.0};

        auto Y = Tensor::matrixmul(X, W);

        std::cout << "Forward Y:" << std::endl;
        Y->print();

        Y->backward();

        std::cout << "\nGradient of W:" << std::endl;
        W->get_grad()->print();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
