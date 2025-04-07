#ifndef BATCHNORM_H
#define BATCHNORM_H

#define EPSILON 1e-8

#include "matrix.h"
#include "optimizer.h"

class BatchNormLayer {
  public:
    std::vector<Matrix> gamma, beta, grad_gamma, grad_beta, EMA_mean, EMA_var, input;
    std::unique_ptr<Optimizer> gamma_optimizer, beta_optimizer;
    double momentum;
    bool training;

    BatchNormLayer(int num_channels, std::unique_ptr<Optimizer> opt, double momentum = 0.9) : momentum(momentum), training(true) {
        gamma.resize(num_channels, Matrix(1, 1, 1.0));
        beta.resize(num_channels, Matrix(1, 1, 0.0));
        grad_gamma.resize(num_channels, Matrix(1, 1));
        grad_beta.resize(num_channels, Matrix(1, 1));
        EMA_mean.resize(num_channels, Matrix(1, 1, 0.0));
        EMA_var.resize(num_channels, Matrix(1, 1, 1.0));
        gamma_optimizer = std::move(opt->clone());
        beta_optimizer = std::move(opt->clone());
    }

    std::vector<Matrix> forward(const std::vector<Matrix>& inputs) {
        this -> input = inputs;
        std::vector<Matrix> outputs(inputs.size());

        for (int c = 0; c < inputs.size(); c++) {
            outputs[c] = Matrix(inputs[c].rows, inputs[c].cols);
            if (training) {
                double mean = 0.0;
                for (int i = 0; i < inputs[c].rows; i++) {
                    for (int j = 0; j < inputs[c].cols; j++) {
                        mean += inputs[c].data[i][j];
                    }
                }
                mean /= (inputs[c].rows * inputs[c].cols);

                double var = 0.0;
                for (int i = 0; i < inputs[c].rows; i++) {
                    for (int j = 0; j < inputs[c].cols; j++) {
                        var += std::pow(inputs[c].data[i][j] - mean, 2);
                    }
                }
                var /= (inputs[c].rows * inputs[c].cols);

                for (int i = 0; i < inputs[c].rows; i++) {
                    for (int j = 0; j < inputs[c].cols; j++) {
                        outputs[c].data[i][j] = gamma[c].data[0][0] * (inputs[c].data[i][j] - mean) / std::sqrt(var + EPSILON) + beta[c].data[0][0];
                    }
                }

                EMA_mean[c].data[0][0] = momentum * EMA_mean[c].data[0][0] + (1 - momentum) * mean;
                EMA_var[c].data[0][0] = momentum * EMA_var[c].data[0][0] + (1 - momentum) * var;

            } else {
                for (int i = 0; i < inputs[c].rows; i++) {
                    for (int j = 0; j < inputs[c].cols; j++) {
                        outputs[c].data[i][j] = gamma[c].data[0][0] * (inputs[c].data[i][j] - EMA_mean[c].data[0][0]) / std::sqrt(EMA_var[c].data[0][0] + EPSILON) + beta[c].data[0][0];
                    }
                }
            }
        }
        return outputs;
    }

    std::vector<Matrix> backward(const std::vector<Matrix>& grad_out) {
        std::vector<Matrix> grad_input(input.size(), Matrix(input[0].rows, input[0].cols));
        int N = input[0].rows * input[0].cols;

        for (int c = 0; c < input.size(); c++) {
            double mean = 0.0;
            for (int i = 0; i < input[c].rows; i++) {
                for (int j = 0; j < input[c].cols; j++) {
                    mean += input[c].data[i][j];
                }
            }
            mean /= N;

            double var = 0.0;
            for (int i = 0; i < input[c].rows; i++) {
                for (int j = 0; j < input[c].cols; j++) {
                    var += std::pow(input[c].data[i][j] - mean, 2);
                }
            }
            var /= N;

            Matrix x_hat(input[c].rows, input[c].cols);
            for (int i = 0; i < input[c].rows; i++) {
                for (int j = 0; j < input[c].cols; j++) {
                    x_hat.data[i][j] = (input[c].data[i][j] - mean) / std::sqrt(var + EPSILON);
                }
            }

            grad_gamma[c].data[0][0] = 0.0;
            grad_beta[c].data[0][0] = 0.0;
            for (int i = 0; i < input[c].rows; i++) {
                for (int j = 0; j < input[c].cols; j++) {
                    grad_gamma[c].data[0][0] += grad_out[c].data[i][j] * x_hat.data[i][j];
                    grad_beta[c].data[0][0] += grad_out[c].data[i][j];
                }
            }

            Matrix grad_x_hat = grad_out[c] * gamma[c].data[0][0];
            double grad_var = 0.0;
            for (int i = 0; i < input[c].rows; i++) {
                for (int j = 0; j < input[c].cols; j++) {
                    grad_var += grad_x_hat.data[i][j] * (input[c].data[i][j] - mean);
                }
            }
            grad_var *= -0.5 * std::pow(var + EPSILON, -1.5);

            double grad_mean = 0.0;
            for (int i = 0; i < input[c].rows; i++) {
                for (int j = 0; j < input[c].cols; j++) {
                    grad_mean += grad_x_hat.data[i][j];
                }
            }
            grad_mean *= -1.0 / std::sqrt(var + EPSILON);

            for (int i = 0; i < input[c].rows; i++) {
                for (int j = 0; j < input[c].cols; j++) {
                    grad_input[c].data[i][j] = (grad_x_hat.data[i][j] / std::sqrt(var + EPSILON)) + (2.0 * grad_var * (input[c].data[i][j] - mean) / N) + (grad_mean / N);
                }
            }
        }
        return grad_input;
    }

    void update() {
        for (int c = 0; c < gamma.size(); c++) {
            gamma_optimizer->update(gamma[c], grad_gamma[c]);
            beta_optimizer->update(beta[c], grad_beta[c]);
        }
    }

    void set_training(bool mode) {
        training = mode;
    }
};

#endif