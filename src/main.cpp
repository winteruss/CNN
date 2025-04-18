#include <iostream>
#include <memory>

#include "util.h"
#include "training.h"
#include "model.h"
#include "dataset.h"

int main() {
    Dataset train_data, test_data;
    train_data.loadCSV("C:\\Users\\saeol\\Desktop\\C Projects\\CNN\\data\\mnist_train_10k.csv", 28, 28, 10);
    test_data.loadCSV("C:\\Users\\saeol\\Desktop\\C Projects\\CNN\\data\\mnist_test.csv", 28, 28, 10);

    int dataset_normal_type = 1;    // 0: Normalize, 1: Standardize

    if (dataset_normal_type == 0) {
        train_data.normalize_dataset();
        test_data.normalize_dataset();
    } else {
        train_data.standardize_dataset();
        test_data.standardize_dataset();
    }

    int num_conv_layers = 2;
    int epochs = 100;
    double lr = 0.0001;

    auto sgd = std::make_unique<SGD>(lr);
    auto momentum = std::make_unique<Momentum>(lr, 0.9);
    auto adagrad = std::make_unique<AdaGrad>(lr);
    auto rmsprop = std::make_unique<RMSProp>(lr, 0.9);
    auto adam = std::make_unique<Adam>(lr, 0.9, 0.999);

    int init_type = 1;  // 0: Set all initial param to 0, 1: He, 2: LeCun

    Model model(28, 28, 10, lr, num_conv_layers, std::move(adam), init_type, 8);

    model.set_training(true);   // Set training mode for batch normalization
    trainDataset(model, train_data, epochs, 0.1);
    model.save("trained_model.txt");

    model.set_training(false);  // Set evaluation mode for batch normalization
    int correct = 0;
    for (size_t i = 0; i < test_data.size(); i++) {
        const auto& [input, target] = test_data[i];
        auto [logits, loss] = model.forward(input, target);
        Matrix probs = softMax(logits);
        int guess = argmax(probs);
        int label = argmax(target);
        if (guess == label) correct++;
    }
    std::cout << "\nAccuracy: " << correct << "/" << test_data.size() << " (" << (static_cast<double>(correct) / test_data.size()) * 100 << "%)\n";

    return 0;
}