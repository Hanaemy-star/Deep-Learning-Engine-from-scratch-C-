#include "tensor.hpp"

int main() {
    try {
        auto a = std::make_shared<Tensor>(std::vector<size_t>{1}, 2.0, true);
        auto b = std::make_shared<Tensor>(std::vector<size_t>{1}, 3.0, true);

        auto c = Tensor::add(a, b);

        auto d = Tensor::add(c, a);

        std::cout << "Forward pass result (d): " << d->get_data()[0] << std::endl;

        d->backward();

        std::cout << "Gradient of a (Expected 2.0): " << a->get_grad()->get_data()[0] << std::endl;
        std::cout << "Gradient of b (Expected 1.0): " << b->get_grad()->get_data()[0] << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
    }

    return 0;
}
