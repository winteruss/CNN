#ifndef FC_H
#define FC_H

#include <memory>

#include "optimizer.h"
#include "matrix.h"

class FCLayer { 
  public:
    Matrix weights, grad_weights;
    Matrix bias, grad_bias;
    Matrix input;   // Save for backpropagation
    std::unique_ptr<Optimizer> weights_optimizer, bias_optimizer;
    int init_type; // 0: random, 1: He, 2: LeCun

    FCLayer(int input_size, int output_size, std::unique_ptr<Optimizer> opt, int init = 0)
    : weights(Matrix(output_size, input_size)),
      grad_weights(Matrix(output_size, input_size)),
      bias(Matrix(output_size, 1)),
      grad_bias(Matrix(output_size, 1)),
      weights_optimizer(std::move(opt -> clone())),
      bias_optimizer(std::move(opt -> clone())), init_type(init) {
        initialize(input_size);
    }

    void initialize(int fan_in) {
        if (init_type == 0) weights.random_init();    // random
        else if (init_type == 1) weights.he_init(fan_in);    // He
        else if (init_type == 2) weights.lecun_init(fan_in);    // LeCun
        bias.random_init(-0.01, 0.01);  
    }

    Matrix forward(const Matrix& input) {    // Assume input is already flattened(column vector)
        this -> input = input;
        Matrix result = weights * input + bias;
        return result;
    }

    Matrix backward(const Matrix& grad_output) {
        // Gradients w.r.t. weights, bias, and input
        grad_weights = grad_output * input.transpose();
        grad_bias = grad_output;
        Matrix grad_input = weights.transpose() * grad_output;
        return grad_input;
    }

    void update() {
        weights_optimizer -> update(weights, grad_weights);
        bias_optimizer -> update(bias, grad_bias);
    }
};

#endif