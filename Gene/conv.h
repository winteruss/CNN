#ifndef CONV_H
#define CONV_H

#include <memory>

#include "optimizer.h"
#include "matrix.h"

class ConvLayer {
  public:
    Matrix kernel, grad_kernel;
    double bias, grad_bias;
    Matrix input;   // Save for backpropagation
    std::unique_ptr<Optimizer> kernel_optimizer, bias_optimizer;
    int init_type; // 0: random, 1: He, 2: LeCun

    ConvLayer(std::unique_ptr<Optimizer> opt, int fan_in, int init=0) : kernel(3, 3), grad_kernel(3, 3), bias(0.0), grad_bias(0.0),
        kernel_optimizer(std::move(opt -> clone())), bias_optimizer(std::move(opt -> clone())), init_type(init) {
        initialize(fan_in);
    }

    void initialize(int fan_in) {
        if (init_type == 0) kernel.random_init();    // random
        else if (init_type == 1) kernel.he_init(fan_in);    // He
        else if (init_type == 2) kernel.lecun_init(fan_in);    // LeCun
        bias = 0.0;
    }

    ConvLayer(const Matrix& k) : kernel(k) {}

    Matrix forward(const Matrix& input) {
        this -> input = input;
        Matrix padded_input = input.pad(kernel.cols / 2);   // Suppose that kernel is square
        Matrix output = padded_input.correlate(kernel) + bias;
        return output;
    }

    Matrix backward(const Matrix& grad_out) {
        Matrix padded_input = input.pad(kernel.cols / 2);
        Matrix grad_input(input.rows, input.cols);
        grad_kernel = Matrix(kernel.rows, kernel.cols);
        grad_bias = 0.0;

        // Gradients w.r.t. kernel, bias and input
        grad_kernel = padded_input.correlate(grad_out);
        for (int i = 0; i < input.rows; i++) {
            for (int j = 0; j < input.cols; j++) {
                grad_bias += grad_out.data[i][j];
            }
        }
        grad_input = grad_out.correlate(kernel, true);
        return grad_input;
    }

    void update() {
        kernel_optimizer -> update(kernel, grad_kernel);
        bias_optimizer -> update(bias, grad_bias);
    }
};

#endif