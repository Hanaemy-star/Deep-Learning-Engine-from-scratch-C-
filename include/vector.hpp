#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <initializer_list>

class Vector {
private:
    std::vector<double> data;

public:
    Vector(size_t size);

    Vector(std::initializer_list<double> list);

    std::vector<double> getData() const;

    size_t size() const;

    double& operator[](size_t i);
    const double& operator[](size_t i) const;

    void print() const;

    Vector operator+(const Vector& other) const;

    Vector operator*(double scalar) const;

    double dot(const Vector& other) const;

    double norm() const;

    double cos_to(const Vector& other) const;
};

