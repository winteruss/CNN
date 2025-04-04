#ifndef TRAIN_H
#define TRAIN_H

#include <iostream>
#include <chrono>

#include "matrix.h"
#include "dataset.h"
#include "model.h"

void trainImage(Model& model, const Matrix& input, const Matrix& target, int epochs) {
    Matrix normalized_input = input.normalize();
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "Epoch " << epoch + 1 << std::endl;

        auto [logits, loss] = model.forward(normalized_input, target);
        model.backward(logits, target);
        model.update();

        std::cout << "  Loss: " << loss << std::endl;
    }

}

void trainDataset(Model& model, Dataset& dataset, int epochs, double target_loss = 0.1) {
    auto start_total = std::chrono::high_resolution_clock::now(); // system time start
    int convergence_epoch = -1; // convergence epoch reset
    for (int epoch = 0; epoch < epochs; epoch++) {
        std::cout << "Epoch " << epoch + 1 << std::endl;
        double total_loss = 0.0;
        dataset.shuffle();
        auto start_epoch = std::chrono::high_resolution_clock::now(); // epoch time start

        for (int i = 0; i < dataset.size(); i++) {
            const auto& [input, target] = dataset[i];
            auto [logits, loss] = model.forward(input, target);
            model.backward(logits, target);
            model.update();
            total_loss += loss;
        }

        auto end_epoch = std::chrono::high_resolution_clock::now(); // epoch time end
        double epoch_time = std::chrono::duration<double>(end_epoch - start_epoch).count();

        double avg_loss = total_loss / dataset.size();
        std::cout << "  Avg Loss: " << avg_loss << ", Epoch Time: " << epoch_time << " seconds" << std::endl;

        if (avg_loss < target_loss && convergence_epoch == -1) {
            convergence_epoch = epoch + 1;
        }
    }

    auto end_total = std::chrono::high_resolution_clock::now(); // system time end
    double total_time = std::chrono::duration<double>(end_total - start_total).count();
    std::cout << "Total Training Time: " << total_time << " seconds" << std::endl;

    if (convergence_epoch == -1) {
        std::cout << "No convergence reached within " << epochs << " epochs." << std::endl;
    } else {
        std::cout << "Final Convergence Epoch: " << convergence_epoch << std::endl;
    }
}

#endif
