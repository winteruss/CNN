#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

class Matrix {
  public:
    std::vector<std::vector<double>> data;
    int rows, cols;

    Matrix() : rows(0), cols(0) {}
    
    Matrix(int r, int c) : rows(r), cols(c), data(r, std::vector<double>(c, 0)) {}
    
    Matrix(std::vector<std::vector<double>> d) : rows(d.size()), cols(d[0].size()), data(d) {}

    Matrix(int r, int c, double val) : rows(r), cols(c), data(r, std::vector<double>(c, val)) {}

    Matrix(int r, int c, std::vector<double>& values) : rows(r), cols(c), data(r, std::vector<double>(c, 0)) {
        if (values.size() != r * c) throw std::invalid_argument("Invalid number of values");
        for (int i = 0; i < r; i++){
            for (int j = 0; j < c; j++){
                data[i][j] = values[i * c + j];
            }
        }
    }

    void print() const {
        for (const auto& row : data) {
            for (double val : row) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
        }
    }

    Matrix pad(int pads) const {
        int output_rows = rows + 2 * pads;
        int output_cols = cols + 2 * pads;

        Matrix padded(output_rows, output_cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                padded.data[i+pads][j+pads] = data[i][j];
            }
        }
        return padded;
    }

    Matrix flatten() const {
        Matrix flat(rows * cols, 1);
        int index = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flat.data[index][0] = data[i][j];
                index++;
            }
        }
        return flat;
    }

    Matrix transpose() const {
        Matrix transposed(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed.data[j][i] = data[i][j];
            }
        }
        return transposed;
    }

    Matrix flip() const {
        Matrix flipped(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flipped.data[i][j] = data[rows - i - 1][cols - j - 1];
            }
        }
        return flipped;
    }

    Matrix normalize() const {    // Min-Max Normalization - Currently not in use
        Matrix normalized(rows, cols);
        double max_val = data[0][0];
        double min_val = data[0][0];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                max_val = std::max(max_val, data[i][j]);
                min_val = std::min(min_val, data[i][j]);
            }
        }

        double range = max_val - min_val;
        if (range == 0.0) range = 1.0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                normalized.data[i][j] = (data[i][j] - min_val) / range;
            }
        }
        return normalized;
    }

    Matrix standardize() const {    // Z-Score Standardization - Currently not in use
        Matrix standardized(rows, cols);
        double mean = 0.0, variance = 0.0;

        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) mean += data[i][j];
        mean /= (rows * cols);

        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) {
            double diff = data[i][j] - mean;
            variance += diff * diff;
        }
        variance /= (rows * cols);
        double std_dev = std::sqrt(variance + 1e-8);

        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) {
            standardized.data[i][j] = (data[i][j] - mean) / std_dev;
        }
        return standardized;
    }

    void randomize(double min_val = -0.5, double max_val = 0.5) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(min_val, max_val);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = dist(gen);
            }
        }
    }

    void he_init(int fan_in) {    // He Initialization
        static std::random_device rd;
        static std::mt19937 gen(rd());
        double std_dev = std::sqrt(2.0 / fan_in);
        std::normal_distribution<double> dist(0.0, std_dev);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = dist(gen);
            }
        }
    }

    void lecun_init(int fan_in) {    // LeCun Initization
        static std::random_device rd;
        static std::mt19937 gen(rd());
        double std_dev = std::sqrt(1.0 / fan_in);
        std::normal_distribution<double> dist(0.0, std_dev);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = dist(gen);
            }
        }
    }

    Matrix correlate(const Matrix& kernel, bool flip_kernel = false) const {
        if (rows < kernel.rows || cols < kernel.cols) {
            throw std::invalid_argument("Input matrix must be larger than or equal to kernel size");
        }

        int out_rows = rows - kernel.rows + 1;
        int out_cols = cols - kernel.cols + 1;
        Matrix result(out_rows, out_cols);

        Matrix _kernel = flip_kernel ? kernel.flip() : kernel;
        
        for (int i = 0; i < out_rows; i++) {
            for (int j = 0; j < out_cols; j++) {
                double sum = 0.0;
                for (int ki = 0; ki < _kernel.rows; ki++) {
                    for (int kj = 0; kj < _kernel.cols; kj++) {
                        sum += data[i + ki][j + kj] * _kernel.data[ki][kj];
                    }
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    Matrix hadamard_power(double power) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = std::pow(data[i][j], power);
            }
        }
        return result;
    }

    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Matrix size mismatch.");
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    }

    Matrix operator+(const double scalar) const {  // Addition with Broadcasting
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] + scalar;
            }
        }
        return result;
    }

    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Matrix size mismatch.");
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] - other.data[i][j];
            }
        }
        return result;
    }

    Matrix operator-(const double scalar) const {  // Subtraction with Broadcasting
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] - scalar;
            }
        }
        return result;
    }

    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) throw std::invalid_argument("Incompatible matrix dimensions.");

        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                for (int k = 0; k < cols; k++) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }

    Matrix operator*(double scalar) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this->data[i][j] * scalar;
            }
        }
        return result;
    }

    friend Matrix operator*(double scalar, const Matrix& mat) {
        return mat * scalar;
    }

    Matrix operator%(const Matrix& other) const {   // Hadamard Multiplication
        if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Matrix size mismatch.");
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = data[i][j] * other.data[i][j];
            }
        }
        return result;
    }

    Matrix operator/(double scalar) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this->data[i][j] / scalar;
            }
        }
        return result;
    }

    Matrix operator/(const Matrix& other) const {   // Hadamard Division
        if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Matrix size mismatch.");
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (other.data[i][j] == 0) throw std::invalid_argument("Division by zero.");
                result.data[i][j] = data[i][j] / other.data[i][j];
            }
        }
        return result;
    }

    friend Matrix operator/(double scalar, const Matrix& other) {   // Hadamard Division
        Matrix result(other.rows, other.cols);
        for (int i = 0; i < other.rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                if (other.data[i][j] == 0) throw std::invalid_argument("Division by zero.");
                result.data[i][j] = scalar / other.data[i][j];
            }
        }
        return result;
    }

    Matrix& operator+=(const Matrix& other) {
        if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Matrix size mismatch.");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] += other.data[i][j];
            }
        }
        return *this;
    }

    Matrix& operator+=(double scalar) {  // Addition with Broadcasting
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] += scalar;
            }
        }
        return *this;
    }

    Matrix& operator-=(const Matrix& other) {
        if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Matrix size mismatch.");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] -= other.data[i][j];
            }
        }
        return *this;
    }

    Matrix& operator-=(double scalar) {  // Subtraction with Broadcasting
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] -= scalar;
            }
        }
        return *this;
    }

    Matrix& operator*=(const Matrix& other) {
        if (cols != other.rows) throw std::invalid_argument("Incompatible matrix dimensions.");

        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                for (int k = 0; k < cols; k++) {
                    result.data[i][j] = data[i][k] * other.data[k][j];
                }
            }
        }
        *this = result;
        return *this;
    }

    Matrix& operator*=(double scalar) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                this->data[i][j] *= scalar;
            }
        }
        return *this;
    }

    Matrix& operator/=(double scalar) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                this->data[i][j] /= scalar;
            }
        }
        return *this;
    }

    Matrix& operator%=(const Matrix& other) {   // Hadamard Multiplication
        if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Matrix size mismatch.");

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] *= other.data[i][j];
            }
        }
        return *this;
    }

    Matrix& operator/=(const Matrix& other) {   // Hadamard Division
        if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Matrix size mismatch.");
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (other.data[i][j] == 0) throw std::invalid_argument("Division by zero.");
                data[i][j] /= other.data[i][j];
            }
        }
        return *this;
    }
};

#endif