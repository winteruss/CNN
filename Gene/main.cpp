#include <iostream>
#include <memory>

#include "util.h"
#include "training.h"
#include "model.h"
#include "dataset.h"
#include "optimizer.h"

int main() {
    Dataset train_data, test_data;

    train_data.loadCSV("/Users/Gene/Desktop/folder_c/c++/CNN_pr/CNN/data/mnist_train_10k.csv", 28, 28, 10);
    test_data.loadCSV("/Users/Gene/Desktop/folder_c/c++/CNN_pr/CNN/data/mnist_test.csv", 28, 28, 10);

    int norm_type = 0; // 0: Min-Max, 1: Z-Score
    train_data.normalize_dataset(norm_type);
    test_data.normalize_dataset(norm_type);

    train_data.shuffle();
    test_data.shuffle();

    int num_conv_layers = 2;
    int fc_input_size = Model::calculate_fc_input_size(28, 28, num_conv_layers);
    int epochs = 100;
    double lr = 0.001;

    auto sgd = std::make_unique<SGD>(lr);
    auto momentum = std::make_unique<Momentum>(lr, 0.9);
    auto adagrad = std::make_unique<AdaGrad>(lr);
    auto rmsprop = std::make_unique<RMSProp>(lr, 0.9);
    auto adam = std::make_unique<Adam>(lr, 0.9, 0.999);

    int init_type = 1; // 0: Random, 1: He, 2: LeCun
    int batch_norm_flag = 1; // 0: deactivate, 1: activate

    Model model(28, 28, 10, lr, num_conv_layers, std::move(adam), batch_norm_flag);

    std::cout << "Initializing convolutional layers..." << std::endl;
    for (int i = 0; i < num_conv_layers; i++) {
        std::cout << "Conv Layer " << i << ": fan_in = " << (3 * 3 * (i > 0 ? 32 : 1)) << std::endl;
        model.conv_layers[i].init_type = init_type;
        model.conv_layers[i].initialize(3 * 3 * (i > 0 ? 32 : 1)); //calculate fan_in
    }
    std::cout << "Fully Connected Layer: input size = " << fc_input_size << std::endl;

   model.fc.init_type = init_type;
   model.fc.initialize(fc_input_size);

    trainDataset(model, train_data, epochs, 0.1);

    model.save("trained_model.txt");

    //std::cout << "\nFinal Predictions:\n";
    for (size_t i = 0; i < train_data.size(); i++) {
        const auto& [input, target] = train_data[i];
        Matrix norm_input = input.min_max_normalize();
        auto [logits, loss] = model.forward(norm_input, target);
        Matrix probs = softMax(logits);
        //std::cout << "Image " << i + 1 << " Loss: " << loss << "\nPredictions:\n";
        //probs.print();
    }

    //std::cout << "\nGuesses:\n";
    for (size_t i = 0; i < train_data.size(); i++) {
        const auto& [input, target] = train_data[i];
        Matrix norm_input = input.min_max_normalize();
        auto [logits, loss] = model.forward(norm_input, target);
        Matrix probs = softMax(logits);
        int guess = argmax(probs);
        int label = argmax(target);
        //std::cout << "Image " << i + 1 << " | Guess: " << guess << " Label: " << label << "\n";
    }

    int correct = 0;
    for (size_t i = 0; i < train_data.size(); i++) {
        const auto& [input, target] = train_data[i];
        Matrix norm_input = input.min_max_normalize();
        auto [logits, loss] = model.forward(norm_input, target);
        Matrix probs = softMax(logits);
        int guess = argmax(probs);
        int label = argmax(target);
        if (guess == label) correct++;
    }
    std::cout << "\nAccuracy: " << correct << "/" << train_data.size() << " (" << (static_cast<double>(correct) / train_data.size()) * 100 << "%)\n";

    return 0;
}