#pragma once

#include "vector.hpp"
#include <iostream>
#include <vector>
#include <stdexcept>

class Matrix {
private:
    size_t rows, cols;
    std::vector<double> data;
    int reverse = 0;

public:
    Matrix(size_t r, size_t c);

    double& operator()(size_t r, size_t c);

    const double& operator()(size_t r, size_t c) const;

    size_t getRows() const;
    size_t getCols() const;

    Matrix transpose() const;

    Vector multiply(const Vector& v) const;

    Matrix operator*(const Matrix& other) const;

    Matrix gaussStairs();

    Matrix& operator=(const Vector& v);

    Matrix& operator=(const Matrix& other);

    Matrix(const Matrix& other);

    double det();

    double trace() const;

    Matrix inverse();

    void print() const;
};