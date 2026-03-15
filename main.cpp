#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <iomanip>
#include "tensor.hpp"
#include "optimizer.hpp"

// Function to generate synthetic data: y = 3x + 2 + noise
void generate_data(size_t num_samples, std::shared_ptr<Tensor>& X, std::shared_ptr<Tensor>& y) {
    X = std::make_shared<Tensor>(std::vector<size_t>{num_samples, 1}, 0.0, false);
    y = std::make_shared<Tensor>(std::vector<size_t>{num_samples, 1}, 0.0, false);

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist_x(-10.0, 10.0);
    std::normal_distribution<double> dist_noise(0.0, 1.0);

    for (size_t i = 0; i < num_samples; ++i) {
        double x_val = dist_x(gen);
        double noise = dist_noise(gen);
        double y_val = 3.0 * x_val + 2.0 + noise;

        X->get_data()[i] = x_val;
        y->get_data()[i] = y_val;
    }
}

int main() {
    std::cout << "=== Starting Neural Network Training ===" << std::endl;

    // 1. Data Generation
    const size_t num_samples = 100;
    std::shared_ptr<Tensor> X_all, y_all;
    generate_data(num_samples, X_all, y_all);
    std::cout << "Generated " << num_samples << " data samples." << std::endl;

    // 2. Model Initialization (1x1 tensors to avoid shape mismatches)
    // Initial Weight W = 0.5, Bias B = 0.0
    auto W = std::make_shared<Tensor>(std::vector<size_t>{1, 1}, 0.5, true);
    auto B = std::make_shared<Tensor>(std::vector<size_t>{1, 1}, 0.0, true);

    std::cout << "Initial Parameters: W = " << W->get_data()[0]
              << ", B = " << B->get_data()[0] << std::endl;

    // 3. Optimizer Setup (Stochastic Gradient Descent)
    double learning_rate = 0.01;
    auto optimizer = std::make_shared<Optimizer>(std::vector<std::shared_ptr<Tensor>>{W, B}, learning_rate);

    // 4. Training Loop
    const int epochs = 100;
    std::cout << "\n--- Starting Training Loop (" << epochs << " epochs) ---" << std::endl;
    std::cout << std::setw(8) << "Epoch" << " | "
              << std::setw(12) << "Avg Loss" << " | "
              << std::setw(10) << "W (curr)" << " | "
              << std::setw(10) << "B (curr)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double epoch_loss = 0.0;

        for (size_t i = 0; i < num_samples; ++i) {
            // Reset gradients for each sample (Stochastic Descent)
            optimizer->zero_grad();

            // Prepare single sample as 1x1 tensors
            auto x_single = std::make_shared<Tensor>(std::vector<size_t>{1, 1}, X_all->get_data()[i], false);
            auto y_single = std::make_shared<Tensor>(std::vector<size_t>{1, 1}, y_all->get_data()[i], false);

            // Step 1: Forward Pass (y_pred = X*W + B)
            auto z = Tensor::matrixmul(x_single, W);
            auto y_pred = Tensor::add(z, B);

            // Step 2: Loss Calculation
            auto loss = Tensor::mse_loss(y_pred, y_single);
            epoch_loss += loss->get_data()[0];

            // Step 3: Backward Pass
            loss->backward();

            // Step 4: Optimizer Update
            optimizer->step();
        }

        // Monitoring progress every 10 epochs
        if (epoch % 10 == 0 || epoch == 1) {
            std::cout << std::setw(8) << epoch << " | "
                      << std::setw(12) << std::fixed << std::setprecision(4) << epoch_loss / num_samples << " | "
                      << std::setw(10) << W->get_data()[0] << " | "
                      << std::setw(10) << B->get_data()[0] << std::endl;
        }
    }

    std::cout << "\n=== Training Complete ===" << std::endl;
    std::cout << "Final Parameters: W = " << W->get_data()[0]
              << ", B = " << B->get_data()[0] << std::endl;
    std::cout << "Target Values:    W = 3.0, B = 2.0" << std::endl;

    return 0;
}