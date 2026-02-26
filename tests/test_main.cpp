#include "tensor.hpp"

void test_transpose() {
    auto a = std::make_shared<Tensor>(std::vector<size_t>{2, 3});
    a->get_data() = {1, 2, 3, 4, 5, 6};

    std::cout << "Original 2x3 matrix:" << std::endl;
    a->print();

    auto b = a->transpose();

    std::cout << "\nTransposed 3x2 matrix:" << std::endl;
    b->print();
}

int main() {
    try {
        test_transpose();
    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
    }

    return 0;
}
