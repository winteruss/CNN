#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>

#include "matrix.h"

class Dataset {
  public:
    std::vector<std::pair<Matrix, Matrix>> data;
    Dataset() = default;

    void loadCSV(const std::string& filename, int rows, int cols, int output_size) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Unable to open: " << filename << std::endl;
        }
        std::string line;
        while (std::getline(file, line)) {
            std::vector<double> values;
            std::stringstream ss(line);
            std::string value;

            while (std::getline(ss, value, ',')) {
                values.push_back(std::stod(value));
            }

            if (values.size() != rows * cols + 1) {
                throw std::runtime_error("CSV row size mismatch: expected " + std::to_string(rows * cols + 1) + ", got " + std::to_string(values.size()));
            }

            Matrix image(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    image.data[i][j] = values[1 + i * cols + j];    // Offset 1 for Label
                }
            }

            int label = static_cast<int>(values[0]);
            Matrix target(output_size, 1);
            target.data[label][0] = 1.0;

            data.emplace_back(image, target);
        }

        file.close();
        std::cout << "Loaded " << data.size() << " image-target pairs from " << filename << "\n";
    }

    size_t size() const {
        return data.size();
    }

    const std::pair<Matrix, Matrix>& operator[](size_t i) const {
        return data[i];
    }

    void normalize_dataset(int norm_type = 0) {    // 0: Min-Max, 1: Z-Score
        if (data.empty()) return;

        std::vector<double> all_values;
        for (const auto& [image, target] : data) {
            for (int i = 0; i < image.rows; i++) {
                for (int j = 0; j < image.cols; j++) {
                    all_values.push_back(image.data[i][j]);
                }
            }
        }

        double min_val = *std::min_element(all_values.begin(), all_values.end());
        double max_val = *std::max_element(all_values.begin(), all_values.end());
        double mean = 0.0, variance = 0.0;

        if (norm_type == 1) {    // Z-Score
            for (double val : all_values) mean += val;
            mean /= all_values.size();

            for (double val : all_values) {
                double diff = val - mean;
                variance += diff * diff;
            }
            variance /= all_values.size();
            double std_dev = std::sqrt(variance + 1e-8);

            for (auto& [image, target] : data) {
                for (int i = 0; i < image.rows; i++) {
                    for (int j = 0; j < image.cols; j++) {
                        image.data[i][j] = (image.data[i][j] - mean) / std_dev;
                    }
                }
            }
        } 
        else {    // Min-Max
            for (auto& [image, target] : data) {
                for (int i = 0; i < image.rows; i++) {
                    for (int j = 0; j < image.cols; j++) {
                        image.data[i][j] = (image.data[i][j] - min_val) / (max_val - min_val + 1e-8);
                        }
                }
            }
        }
    }

    // data_shuffle
    void shuffle() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(data.begin(), data.end(), gen);
    }
};

#endif