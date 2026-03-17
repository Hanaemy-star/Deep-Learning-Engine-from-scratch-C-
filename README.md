# Self-Made Deep Learning Framework in C++

## Overview

This project is a lightweight, purely custom Deep Learning framework built from scratch in C++. It was created to deeply understand the underlying mathematics and architectural principles of modern libraries like PyTorch or TensorFlow.

The framework successfully solves the non-linear **XOR problem**, proving the correctness of its autograd engine and optimization logic.

## Key Features

* **Custom Tensor Class:** Handles N-dimensional data and manages the computational graph.
* **Automatic Differentiation (Autograd):** Full implementation of backpropagation using dynamic computational graphs and topological sorting.
* **Linear Layer:** Fully connected layer with proper weight initialization (Xavier/Glorot).
* **Activation Functions:** Implemented Leaky ReLU to prevent the "dying ReLU" problem.
* **Loss Function:** Mean Squared Error (MSE) Loss with custom backward pass.
* **Optimizer:** Stochastic Gradient Descent (SGD) for weight updates.
* **Operator Overloading:** Intuitive syntax for model definitions (e.g., `(input * W) + B`).

## Deep Dive: What's Inside?

Writing this framework required implementing several core concepts from scratch:

1.  **Computational Graph:** Every operation (`add`, `matmul`, `leaky_relu`) registers itself and its parent tensors, building a directed acyclic graph (DAG) on the fly.
2.  **Topological Sort:** To compute gradients correctly, the framework sorts the graph so that a tensor's gradient is only computed after all its "children" in the graph have been processed.
3.  **Chain Rule Implementation:** Each operation has a custom `_backward` lambda function that applies the calculus chain rule to propagate gradients backward.
4.  **Memory Management:** Heavy use of `std::shared_ptr` to manage tensor lifetimes within the complex graph structure.

## How to Run

### Prerequisites

* A C++ compiler supporting C++17 or later (e.g., g++, clang).
* CMake (optional, but recommended).

### Compilation (Manual)

```bash
g++ -std=c++17 main.cpp tensor.cpp linear.cpp mlp.cpp optimizer.cpp -o dl_framework
./dl_framework
```
### Authors:
**Igor/Hanaemy**
