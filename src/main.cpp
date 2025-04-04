#include <iostream>
#include <memory>

#include "util.h"
#include "training.h"
#include "model.h"
#include "dataset.h"

int main() {
    Dataset train_data, test_data;
    train_data.loadCSV("C:\\Users\\saeol\\Desktop\\C Projects\\CNN\\data\\mnist_test.csv", 28, 28, 10);
    test_data.loadCSV("C:\\Users\\saeol\\Desktop\\C Projects\\CNN\\data\\mnist_test.csv", 28, 28, 10);

    int dataset_normal_type = 0;    // 0: Normalize, 1: Standardize
    if (dataset_normal_type == 0) {
        train_data.normalize_dataset();
        test_data.normalize_dataset();
    } else {
        train_data.standardize_dataset();
        test_data.standardize_dataset();
    }

    int num_conv_layers = 2;
    int fc_input_size = Model::calculate_fc_input_size(28, 28, num_conv_layers);
    int epochs = 100;
    double lr = 0.1;

    auto sgd = std::make_unique<SGD>(lr);
    auto momentum = std::make_unique<Momentum>(lr, 0.9);
    auto adagrad = std::make_unique<AdaGrad>(lr);
    auto rmsprop = std::make_unique<RMSProp>(lr, 0.9);
    auto adam = std::make_unique<Adam>(lr, 0.9, 0.999);

    int init_type = 1; // 0: Random, 1: He, 2: LeCun
    bool normal_type = true;    // 0: Normalize, 1: Standardize

    Model model(28, 28, 10, lr, num_conv_layers, std::move(adam), init_type);

    trainDataset(model, train_data, epochs, 0.01, normal_type);
    model.save("trained_model.txt");

    //std::cout << "\nFinal Predictions:\n";
    for (size_t i = 0; i < train_data.size(); i++) {
        const auto& [input, target] = train_data[i];
        Matrix norm_input = input.normalize();
        auto [logits, loss] = model.forward(norm_input, target);
        Matrix probs = softMax(logits);
        //std::cout << "Image " << i + 1 << " Loss: " << loss << "\nPredictions:\n";
        //probs.print();
    }

    //std::cout << "\nGuesses:\n";
    for (size_t i = 0; i < train_data.size(); i++) {
        const auto& [input, target] = train_data[i];
        Matrix norm_input = input.normalize();
        auto [logits, loss] = model.forward(norm_input, target);
        Matrix probs = softMax(logits);
        int guess = argmax(probs);
        int label = argmax(target);
        //std::cout << "Image " << i + 1 << " | Guess: " << guess << " Label: " << label << "\n";
    }

    int correct = 0;
    for (size_t i = 0; i < train_data.size(); i++) {
        const auto& [input, target] = train_data[i];
        Matrix norm_input = input.normalize();
        auto [logits, loss] = model.forward(norm_input, target);
        Matrix probs = softMax(logits);
        int guess = argmax(probs);
        int label = argmax(target);
        if (guess == label) correct++;
    }
    std::cout << "\nAccuracy: " << correct << "/" << train_data.size() << " (" << (static_cast<double>(correct) / train_data.size()) * 100 << "%)\n";

    return 0;
}