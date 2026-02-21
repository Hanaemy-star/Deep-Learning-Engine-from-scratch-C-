#include "vector.hpp"

Vector::Vector(size_t size) : data(size, 0.0) {}

Vector::Vector(std::initializer_list<double> list) : data(list) {}

size_t Vector::size() const {return data.size();}

std::vector<double> Vector::getData() const {return data;}

double& Vector::operator[](size_t i) {return data[i];}

const double& Vector::operator[](size_t i) const {return data[i];}

void Vector::print() const {
    std::cout << "(";
    for (size_t i = 0; i < data.size(); i++) {
        std::cout << data[i] << (i == data.size() - 1 ? "" : ", ");
    }
    std::cout << ")" << std::endl;
}

Vector Vector::operator+(const Vector& other) const {
    Vector result(size());
    for (size_t i = 0; i < data.size(); i++) {
        result[i] = data[i] + other[i];
    }
    return result;
}

Vector Vector::operator*(double scalar) const {
    Vector result(size());
    for (size_t i = 0; i< data.size(); i++) {
        result[i] = data[i] * scalar;
    }
    return result;
}

double Vector::dot(const Vector& other) const {
    double result = 0;
    for (size_t i = 0; i < data.size(); i++) {
        result += data[i] * other[i];
    }
    return result;
}

double Vector::norm() const {
    double result = 0;
    return std::sqrt(dot(*this));
}

double Vector::cos_to(const Vector& other) const {
    if (this->size() != other.size()) {
        throw std::invalid_argument("Vector::cos_to: size mismatch");
    }

    double result = 0;
    result = dot(other) / (norm() * other.norm());
    return result;
}
