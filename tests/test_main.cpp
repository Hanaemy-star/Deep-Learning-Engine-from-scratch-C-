#include "tensor.hpp"

int main() {
    try {
        Tensor W({2, 2});
        W({0, 0}) = 0.5;  W({0, 1}) = 2.0;
        W({1, 0}) = -1.0; W({1, 1}) = 0.5;

        Tensor X({2, 1});
        X({0, 0}) = 4.0;
        X({1, 0}) = 1.0;

        std::cout << "Weights W:" << std::endl;
        W.print();
        std::cout << "Input X:" << std::endl;
        X.print();

        Tensor Y = W.matmul(X);
        std::cout << "Result before ReLU (Y = W*X):" << std::endl;
        Y.print();

        Tensor Z = Y.relu();
        std::cout << "Result after ReLU:" << std::endl;
        Z.print();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
