#include "matrix.hpp"


Matrix::Matrix(size_t r, size_t c) : rows(r), cols(c), data(rows * cols, 0.0) {}

Matrix::Matrix(const Matrix& other)
    : rows(other.rows),
      cols(other.cols),
      data(other.data),
      reverse(other.reverse)
{
}

double& Matrix::operator()(size_t r, size_t c) {
    return data[r * cols + c];
}

const double& Matrix::operator()(size_t r, size_t c) const {
    return data[r * cols + c];
}

size_t Matrix::getRows() const {
    return rows;
}

size_t Matrix::getCols() const {
    return cols;
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

Vector Matrix::multiply(const Vector &v) const {
    Vector result(rows);
    Vector temp(cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            temp[j] = data[j + i * cols];
        }
        result[i] = temp.dot(v);
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (this->cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions mismatch: left.cols must equal right.rows");
    }
    Matrix result(rows, other.cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < other.cols; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < other.rows; k++) {
                sum += (*this)(i, k) * other(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

Matrix& Matrix::operator=(const Vector& v) {
    if (this->rows * this->cols == v.getData().size()) {
        this->data = v.getData();
    } else {
        this->data = v.getData();
        this->rows = 1;
        this->cols = this->data.size();
    }
    return *this;
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this == &other) return *this;
    data = other.data;
    rows = other.rows;
    cols = other.cols;
    reverse = other.reverse;
    return *this;
}

void Matrix::print() const {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << (*this)(i, j) << " ";
        }
        std::cout << "\n";
    }
}

Matrix Matrix::gaussStairs() {
    Matrix result(*this);
    size_t minimum = std::min(rows, cols);

    for (size_t k = 0; k < minimum; k++) {
        double max_val = std::abs(result(k, k));
        size_t index_max = k;

        for (size_t i = k + 1; i < rows; i++) {
            if (std::abs(result(i, k)) > max_val) {
                max_val = std::abs(result(i, k));
                index_max = i;
            }
        }

        if (max_val < 1e-9) {
            continue;
        }

        if (index_max != k) {
            for (size_t j = 0; j < cols; j++) {
                std::swap(result(k, j), result(index_max, j));
            }
            reverse += 1;
        }

        for (size_t i = k + 1; i < rows; i++) {
            double factor = result(i, k) / result(k, k);
            for (size_t j = k; j < cols; j++) {
                result(i, j) -= factor * result(k, j);
            }
            result(i, k) = 0;
        }
    }
    return result;
}

double Matrix::det() {
    if (rows != cols) {
        return 0;
    } else {
        double result = 1;
        Matrix temp = gaussStairs();
        for (size_t i = 0; i < rows; i++) {
            result *= temp(i, i);
        }
        if (reverse % 2 != 0) {
            result *= -1;
        }
        reverse = 0;
        if (std::abs(result) < 1e-12) {
            return 0;
        }
        return result;
    }
}

Matrix Matrix::inverse() {
    if (rows != cols) {
        throw std::invalid_argument("Matrix inverse: matrix must be square");
    }
    Matrix expand(rows, 2 * cols);
    // добавляем единицы в присоединенную
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            expand(i, j) = (*this)(i, j);
            if (i == j) {
                expand(i, j + cols) = 1.0;
            } else {
                expand(i, j + cols) = 0.0;
            }
        }
    }
    // приводим к ступенчетаму виду
    expand = expand.gaussStairs();
    reverse = 0;
    for (int k = rows - 1; k >= 0; k--) {

        double diag = expand(k, k);
        if (std::abs(diag) < 1e-12) throw std::runtime_error("Matrix is singular");

        for (size_t j = 0; j < 2 * cols; j++) {
            expand(k, j) /= diag;
        }

        for (int i = k - 1; i >= 0; i--) {
            double factor = expand(i, k);
            for (size_t j = k; j < 2 * cols; j++) {
                expand(i, j) -= factor * expand(k, j);
            }
        }
    }
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; j++) {
            result(i, j) = expand(i, j + cols);
        }
    }
    return result;
}

double Matrix::trace() const {
    if (rows == cols) {
        double result = 0;
        for (size_t i = 0; i < rows; i++) {
            result += (*this)(i, i);
        }
        return result;
    }
    throw std::invalid_argument("Matrix trace(): matrix must be square");
}