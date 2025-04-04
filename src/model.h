#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <fstream>
#include <memory>

#include "conv.h"
#include "pool.h"
#include "fc.h"
#include "activation.h"
#include "loss.h"
#include "optimizer.h"

class Model {
  public:
    std::vector<ConvLayer> conv_layers;
    std::vector<PoolLayer> pool_layers;
    FCLayer fc;

    int input_size;
    int output_size;
    double learning_rate;
    int init_type;  // 0: random, 1: He, 2: LeCun

    std::vector<Matrix> intermediates;   // Save for backpropagation

    Model(int input_rows, int input_cols, int output_size, double lr, int num_conv_layers, std::unique_ptr<Optimizer> opt, int init_type)
    // Adjust input size of FC layer regarding pooling layers, assuming 2x2 and stride 2
    : fc(calculate_fc_input_size(input_rows, input_cols, num_conv_layers), output_size, opt -> clone(), init_type), output_size(output_size), learning_rate(lr) {
        conv_layers.reserve(num_conv_layers);
        for (int i = 0; i < num_conv_layers; i++) {
            conv_layers.emplace_back(opt -> clone(), 1 * 5 * 5, init_type); // 1 channel, 5x5 kernel
            pool_layers.emplace_back();
        }
        intermediates.resize(2 * num_conv_layers + 1);  // Conv, ReLu Outputs + FC Input
    }

    std::pair<Matrix, double> forward(const Matrix& input, const Matrix& target) {
        Matrix x = input;
        int idx = 0;
        for (int i = 0; i < conv_layers.size(); i++) {
            x = conv_layers[i].forward(x);
            intermediates[idx++] = x;
            x = leakyReLU(x);
            intermediates[idx++] = x;
            x = pool_layers[i].forward(x);
        }
        intermediates[idx] = x;
        x = x.flatten();
        x = fc.forward(x);
        
        double loss = crossEntropyLoss(softMax(x), target);
        return {x, loss};
    }

    void backward(const Matrix& logits, const Matrix& target) {
        Matrix probs = softMax(logits);
        Matrix grad = probs - target;

        grad = fc.backward(grad);
        grad = pool_layers.back().backward(grad);
        grad = leakyReLU_backward(intermediates[intermediates.size() - 2], grad);   // LeakyReLU input
        for (int i = conv_layers.size() - 1; i >= 0; i--) {
            grad = conv_layers[i].backward(grad);
            if (i > 0) {
                grad = pool_layers[i - 1].backward(grad);
                grad = leakyReLU_backward(intermediates[2 * i - 1], grad);
            }
        }
    }

    void update() {
        for (auto& conv : conv_layers) conv.update();
        fc.update();
    }

    void save(const std::string& filename) const {
        std::ofstream ofs(filename);
        if (!ofs) throw std::runtime_error("Cannot open file for saving.");
        for (const auto& conv : conv_layers) {
            ofs << "Conv Kernel:\n";
            for (int i = 0; i < conv.kernel.rows; i++) {
                for (int j = 0; j < conv.kernel.cols; j++) {
                    ofs << conv.kernel.data[i][j] << " ";
                }
                ofs << "\n";
            }
            ofs << "Conv Bias:\n" << conv.bias << "\n";
        }
        ofs << "FC Weights:\n";
        for (int i = 0; i < fc.weights.rows; i++) {
            for (int j = 0; j < fc.weights.cols; j++) {
                ofs << fc.weights.data[i][j] << " ";
            }
            ofs << "\n";
        }
        ofs << "FC Bias:\n";
        for (int i = 0; i < fc.bias.rows; i++) {
            ofs << fc.bias.data[i][0] << " ";
        }
    }

    static int calculate_fc_input_size(int rows, int cols, int num_conv_layers) {
        int reduced_rows = rows;
        int reduced_cols = cols;
        for (int i = 0; i < num_conv_layers; i++) {
            reduced_rows = reduced_rows / 2;
            reduced_cols = reduced_cols / 2;
        }
        return reduced_rows * reduced_cols;
    }
    
    /* To be implemented...
    void load(const std::string& filename) const {
        std::ifstream ifs(filename);
        if (!ifs) throw std::runtime_error("Cannot open file for loading.");
        std::string line;
        for (size_t i = 0; i < conv_layers.size(); i++) {
            std::getline(ifs, line);

        }
    } */
};

#endif