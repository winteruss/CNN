#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <fstream>
#include <memory>

#include "conv.h"
#include "batchnorm.h"
#include "pool.h"
#include "fc.h"
#include "activation.h"
#include "loss.h"
#include "optimizer.h"

class Model {
  public:
    std::vector<ConvLayer> conv_layers;
    std::vector<BatchNormLayer> bn_layers;
    std::vector<PoolLayer> pool_layers;
    FCLayer fc;

    int output_size;
    double learning_rate;
    int init_type;  // 0: random, 1: He, 2: LeCun
    int num_kernels_per_layer;

    std::vector<std::vector<Matrix>> intermediates;   // Save for backpropagation

    Model(int input_rows, int input_cols, int output_size, double lr, int num_conv_layers, std::unique_ptr<Optimizer> opt, int init_type, int num_kernels = 8)
    // Adjust input size of FC layer regarding pooling layers, assuming 2x2 and stride 2
    : output_size(output_size), learning_rate(lr), init_type(init_type), num_kernels_per_layer(num_kernels),
      fc(calculate_fc_input_size(input_rows, input_cols, num_conv_layers) * num_kernels, output_size, opt->clone(), init_type) {
        conv_layers.reserve(num_conv_layers);
        int num_input_channels = 1; // First layer
        for (int i = 0; i < num_conv_layers; i++) {
            conv_layers.emplace_back(opt -> clone(), num_input_channels, num_kernels, init_type);
            bn_layers.emplace_back(num_kernels, opt -> clone(), 0.9);
            pool_layers.emplace_back();
            num_input_channels = num_kernels; // Output channels become input channels for next layer
        }
        intermediates.resize(3 * num_conv_layers + 1);  // Conv, ReLu Outputs + FC Input
    }

    std::pair<Matrix, double> forward(const Matrix& input, const Matrix& target) {
        std::vector<Matrix> x = {input};    // Start with single-channel input
        int idx = 0;
        for (int i = 0; i < conv_layers.size(); i++) {
            std::vector<Matrix> conv_out = conv_layers[i].forward(x);
            intermediates[idx++] = conv_out;
            std::vector<Matrix> bn_out = bn_layers[i].forward(conv_out);
            intermediates[idx++] = bn_out;
            for (auto& out : bn_out) out = leakyReLU(out);
            intermediates[idx++] = bn_out;
            std::vector<Matrix> pooled(bn_out.size());
            for (size_t j = 0; j < bn_out.size(); j++) {
                pooled[j] = pool_layers[i].forward(bn_out[j]);
            }
            x = pooled;
        }
        intermediates[idx] = x;
        Matrix flat(1, x.size() * x[0].rows * x[0].cols);
        int k = 0;
        for (const auto& mat : x) {
            Matrix f = mat.flatten();
            for (int i = 0; i < f.rows; i++) flat.data[0][k++] = f.data[i][0];
        }
        Matrix logits = fc.forward(flat.transpose());
        double loss = crossEntropyLoss(softMax(logits), target);
        return {logits, loss};
    }

    void backward(const Matrix& logits, const Matrix& target) {
        Matrix probs = softMax(logits);
        Matrix grad = probs - target;   // Gradient of cross-entropy loss w.r.t. logits

        grad = fc.backward(grad);

        // Reshape grad into multi-channel format (e.g., 8 channels of 4x4 for last layer)
        const auto& last_pooled = intermediates.back(); // Shape: [num_filters, pooled_rows, pooled_cols]
        int num_channels = last_pooled.size();
        int channel_size = last_pooled[0].rows * last_pooled[0].cols;
        std::vector<Matrix> grad_x(num_channels, Matrix(last_pooled[0].rows, last_pooled[0].cols));
        int k = 0;
        for (int c = 0; c < num_channels; c++) {
            for (int r = 0; r < last_pooled[0].rows; r++) {
                for (int col = 0; col < last_pooled[0].cols; col++) {
                    grad_x[c].data[r][col] = grad.data[k++][0];
                }
            }
        }

        // Backprop through conv and pool layers
        for (int i = conv_layers.size() - 1; i >= 0; i--) {
            // Backprop through pooling
            for (size_t j = 0; j < grad_x.size(); j++) {
                grad_x[j] = pool_layers[i].backward(grad_x[j]); // Upscale to conv output size
            }

            // Backprop through LeakyReLU
            for (size_t j = 0; j < grad_x.size(); j++) {
                grad_x[j] = leakyReLU_backward(intermediates[3 * i + 1][j], grad_x[j]);
            }

            // Backprop through bn, conv layer
            grad_x = bn_layers[i].backward(grad_x);
            grad_x = conv_layers[i].backward(grad_x); // grad_x now has gradients w.r.t. conv input

            // For all but the first layer, grad_x matches the previous layer's pooled output size
            if (i > 0) {
                grad_x.resize(intermediates[3 * (i - 1) + 1].size(), 
                              Matrix(intermediates[3 * (i - 1)].size() ? intermediates[3 * (i - 1)][0].rows : grad_x[0].rows, 
                                     intermediates[3 * (i - 1)].size() ? intermediates[3 * (i - 1)][0].cols : grad_x[0].cols));
            }
        }
    }

    void update() {
        for (auto& conv : conv_layers) conv.update();
        for (auto& bn : bn_layers) bn.update();
        fc.update();
    }

    void save(const std::string& filename) const {
        std::ofstream ofs(filename);
        if (!ofs) throw std::runtime_error("Cannot open file for saving.");
    
        // Save convolutional layers
        ofs << "Convolutional Layers: " << conv_layers.size() << "\n\n";
        for (size_t layer = 0; layer < conv_layers.size(); layer++) {
            const auto& conv = conv_layers[layer];
            ofs << "Layer " << layer << " Filters: " << conv.num_kernels << "\n";
            ofs << "Input Channels: " << conv.kernels[0].size() << "\n\n";
    
            // Save kernels for each filter and channel
            for (int filter = 0; filter < conv.num_kernels; filter++) {
                ofs << "Filter " << filter << " Kernels:\n";
                for (size_t channel = 0; channel < conv.kernels[filter].size(); channel++) {
                    ofs << "Channel " << channel << ":\n";
                    const Matrix& kernel = conv.kernels[filter][channel];
                    for (int i = 0; i < kernel.rows; i++) {
                        for (int j = 0; j < kernel.cols; j++) {
                            ofs << kernel.data[i][j] << " ";
                        }
                        ofs << "\n";
                    }
                    ofs << "\n";
                }
            }
            ofs << "Bias: " << conv.bias << "\n\n";

            // Save batch normalization layers
            const auto& bn = bn_layers[layer];
            ofs << "Batch Normalization Layer: " << layer << ":\n";
            ofs << "Gamma: ";
            for (const auto& g : bn.gamma) ofs << g.data[0][0] << " ";
            ofs << "\nBeta: ";
            for (const auto& b : bn.beta) ofs << b.data[0][0] << " ";
            ofs << "\nEMA Mean: ";
            for (const auto& m : bn.EMA_mean) ofs << m.data[0][0] << " ";
            ofs << "\nEMA Var: ";
            for (const auto& v : bn.EMA_var) ofs << v.data[0][0] << " ";
            ofs << "\n\n";
        }
    
        // Save fully connected layer
        ofs << "FC Weights: " << fc.weights.rows << "x" << fc.weights.cols << "\n";
        for (int i = 0; i < fc.weights.rows; i++) {
            for (int j = 0; j < fc.weights.cols; j++) {
                ofs << fc.weights.data[i][j] << " ";
            }
            ofs << "\n";
        }
    
        ofs << "\nFC Bias: " << fc.bias.rows << "\n";
        for (int i = 0; i < fc.bias.rows; i++) {
            ofs << fc.bias.data[i][0] << " ";
        }
        ofs << "\n";
    
        ofs.close();
    }

    void set_training(bool mode) {
        for (auto& bn : bn_layers) bn.set_training(mode);
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
};

#endif
