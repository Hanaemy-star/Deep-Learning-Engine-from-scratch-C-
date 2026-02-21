#include "vector.hpp"
#include "matrix.hpp"

int main() {
    // Vector v1 = {1.0, 2.0, 2.0};
    // Vector v2 = {4.0, 0.0, -3.0};
    //
    // std::cout << "Vector 1: "; v1.print();
    // std::cout << "Vector 2: "; v2.print();
    //
    // std::cout << "Dot product: " << v1.dot(v2) << std::endl;
    // std::cout << "Norm v1: " << v1.norm() << std::endl;
    // std::cout << "Norm v2: " << v2.norm() << std::endl;
    // std::cout << "cos: " << v1.cos_to(v2) << std::endl;

    Vector v1 = {1.0, 2.0, 3.0, 4.0};
    Vector v2 = {0.0, 1.0, 1.0, 0.0};
    Vector v3 = {2.0, -1.0, 0.0, 0.0, 3.0, 1.0, 1.0, 0.0, 4.0};
    Vector v4 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    Matrix m1(2, 2);
    Matrix m2(2, 2);
    Matrix m3(3, 3);
    Matrix m4(3, 3);
    m1 = v1;
    m2 = v2;
    m3 = v3;
    m4 = v4;
    std::cout << "det m1: " << m1.det() << std::endl;
    std::cout << "det m2: " << m2.det() << std::endl;
    std::cout << "det m3: " << m3.det() << std::endl;
    std::cout << "det m4: " << m4.det() << std::endl;
    m3.print();
    m3.transpose().print();
    m3.inverse().print();
    (m3*m3.inverse()).print();
    std::cout << m3.trace() << std::endl;
    return 0;
}