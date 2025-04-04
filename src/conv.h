#ifndef CONV_H
#define CONV_H

#include <memory>

#include "optimizer.h"

class ConvLayer {
  public:
    std::vector<std::vector<Matrix>> kernels, grad_kernels;
    double bias, grad_bias;
    std::vector<Matrix> input;   // Save for backpropagation
    std::unique_ptr<Optimizer> kernel_optimizer, bias_optimizer;
    int init_type; // 0: random, 1: He, 2: LeCun
    int num_kernels;

    ConvLayer(std::unique_ptr<Optimizer> opt, int num_input_channels, int num_kernels = 8, int init = 0) : bias(0.0), grad_bias(0.0),
      kernel_optimizer(std::move(opt -> clone())), bias_optimizer(std::move(opt -> clone())), init_type(init), num_kernels(num_kernels) {
        kernels.resize(num_kernels, std::vector<Matrix>(num_input_channels, Matrix(5, 5)));
        grad_kernels.resize(num_kernels, std::vector<Matrix>(num_input_channels, Matrix(5, 5)));
        for (auto& filter_kernels : kernels) { 
            for (auto& kernel : filter_kernels) {
                if (init_type == 0) kernel.randomize();
                else if (init_type == 1) kernel.he_init(num_input_channels * 5 * 5);
                else if (init_type == 2) kernel.lecun_init(num_input_channels * 5 * 5);
            }
        }
    }

    std::vector<Matrix> forward(const std::vector<Matrix>& inputs) {
        this -> input = inputs;
        std::vector<Matrix> outputs(num_kernels);

        for (int i = 0; i < num_kernels; i++) {
            outputs[i] = Matrix(inputs[0].rows, inputs[0].cols); // Initialize output feature map
            for (int c = 0; c < inputs.size(); c++) {
                Matrix padded_input = inputs[c].pad(kernels[0][0].cols / 2);
                outputs[i] += padded_input.correlate(kernels[i][c]); // Sum across channels
            }
            outputs[i] += bias; // Add bias to entire feature map
        }
        return outputs;
    }

    std::vector<Matrix> backward(const std::vector<Matrix>& grad_out) {
        std::vector<Matrix> grad_input(input.size(), Matrix(input[0].rows, input[0].cols));
        grad_kernels = std::vector<std::vector<Matrix>>(num_kernels, std::vector<Matrix>(input.size(), Matrix(5, 5)));
        grad_bias = 0.0;

        int pad_size = kernels[0][0].cols / 2;
        for (int i = 0; i < num_kernels; i++) {
            Matrix padded_grad_out = grad_out[i].pad(pad_size); // Adjust padding
            for (int c = 0; c < input.size(); c++) {
                Matrix padded_input = input[c].pad(pad_size);
                // Gradients w.r.t. kernel and input
                grad_kernels[i][c] = padded_input.correlate(grad_out[i]);
                grad_input[c] += padded_grad_out.correlate(kernels[i][c], true); // Full convolution for input grad
            }
            for (int r = 0; r < grad_out[i].rows; r++) {
                for (int c = 0; c < grad_out[i].cols; c++) {
                    // Gradients w.r.t. bias
                    grad_bias += grad_out[i].data[r][c];
                }
            }
        }
        return grad_input;
    }

    void update() {
        for (int i = 0; i < num_kernels; i++) {
            for (int c = 0; c < kernels[i].size(); c++) {
                kernel_optimizer->update(kernels[i][c], grad_kernels[i][c]);
            }
        }
        bias_optimizer->update(bias, grad_bias);
    }
};

#endif