
# Quantum Annealing Benchmarking and Training for Chatbot Model

## Overview

This repository contains Python scripts for benchmarking and training a simple chatbot model using a simulated quantum annealing approach. The goal of this project is to use quantum-inspired optimization to effectively tune the hyperparameters of the chatbot model and evaluate its performance across different computing devices (CPU and GPU).

### Files in This Repository
- `benchmark.py`: Main entry point for benchmarking the training of the chatbot model. It stages and benchmarks the quantum annealing process to measure the training effectiveness on different devices.
- `main.py`: Script for training the chatbot model and performing inference with some sample inputs. It initializes the training parameters, preprocesses data, and saves the trained model.
- `train.py`: Contains the core implementation of the quantum annealing algorithm used for training the chatbot model, using both multiprocessing for CPU and sequential processing for GPU to optimize the training process.

## Quantum Annealing Algorithm

### Introduction
Quantum annealing is an optimization technique inspired by quantum mechanics. It involves finding the global minimum of a cost function by using a process similar to simulated annealing but utilizing quantum tunneling. In this project, we employ a simulated quantum annealing process to optimize the hyperparameters of the chatbot model, aiming to minimize the loss during training.

### Why Quantum Annealing is Better for CPUs than GPUs
Simulated quantum annealing inherently involves evaluating multiple candidate solutions simultaneously, which makes it suitable for a multiprocessing approach. CPUs, with their numerous cores and the ability to manage multiple threads or processes efficiently, are well-suited for this type of workload.

In contrast, GPUs are designed for highly parallel computations, typically involving vectorized operations on large datasets (e.g., matrix multiplications for neural networks). However, the process of quantum annealing involves iterative optimization with a mix of random perturbations and conditional evaluations, which do not map well to the SIMD (Single Instruction, Multiple Data) architecture of GPUs. Additionally, CUDA, the programming model for GPUs, does not efficiently support running many independent tasks, making the CPU a better fit for the quantum annealing approach.

### Mathematical Description of the Simulated Quantum Annealing Algorithm

Simulated quantum annealing aims to find an optimized solution by probabilistically selecting configurations that minimize an objective function, which is typically the loss function of the model. The process is inspired by the concept of "quantum tunneling," which allows for escaping local minima more effectively than classical annealing.

#### Algorithm Steps
1. **Initialization**:
   - Define the bounds for each hyperparameter.
   - Initialize the current solution \( \mathbf{x}_{	ext{current}} \) randomly within the bounds.
   - Set an initial temperature \( T_0 \), which controls the probability of accepting worse solutions.

2. **Candidate Generation**:
   - For each iteration, generate a new candidate solution \( \mathbf{x}_{	ext{new}} \) by perturbing \( \mathbf{x}_{	ext{current}} \) with a random noise term:
     \[
     \mathbf{x}_{	ext{new}} = \mathbf{x}_{	ext{current}} + \Delta \mathbf{x}
     \]
     where \( \Delta \mathbf{x} \) is drawn from a normal distribution scaled by a step size parameter.

3. **Energy Evaluation**:
   - Calculate the energy (or objective function value) for both \( \mathbf{x}_{	ext{current}} \) and \( \mathbf{x}_{	ext{new}} \). The energy is the value of the loss function for the chatbot model:
     \[
     E(\mathbf{x}) = 	ext{Loss}(\mathbf{x})
     \]

4. **Acceptance Criterion**:
   - If the new candidate has a lower energy (\( E(\mathbf{x}_{	ext{new}}) < E(\mathbf{x}_{	ext{current}}) \)), accept it as the new current solution.
   - If the new candidate has a higher energy, accept it with a probability given by the Metropolis criterion:
     \[
     P_{	ext{accept}} = \exp\left( -rac{E(\mathbf{x}_{	ext{new}}) - E(\mathbf{x}_{	ext{current}})}{T} ight)
     \]
     where \( T \) is the current temperature. This probability allows the algorithm to escape local minima, as it permits worse solutions to be accepted, especially at higher temperatures.

5. **Temperature Update**:
   - The temperature is gradually decreased according to a cooling schedule:
     \[
     T = rac{T_0}{1 + k \cdot t}
     \]
     where \( t \) is the iteration number and \( k \) is a cooling rate constant. This ensures that the acceptance probability of worse solutions decreases over time, focusing the search on exploitation rather than exploration.

6. **Step Size Adaptation**:
   - The step size for generating new candidates is adapted during the optimization process to balance between exploring new regions of the solution space (large step size) and fine-tuning around promising solutions (small step size).

7. **Convergence and Early Termination**:
   - The algorithm monitors the energy values over a sliding window to determine whether the solution has converged.
   - If the relative improvement in energy over a certain number of iterations falls below a threshold, the optimization is terminated early.

### Usage in Training
In the training process (`train.py`), the quantum annealing algorithm is used to optimize the following hyperparameters:
- **Input Dimension**: Dimension of the input feature vector.
- **Hidden Dimension**: Number of hidden units in the model's hidden layer.
- **Learning Rate**: Step size used by the optimizer (Adam).

The `train_model` function in `train.py` leverages multiprocessing for CPU execution, distributing the evaluation of candidate solutions across multiple processes to speed up the optimization. For GPU, the process is executed sequentially due to limitations in concurrent CUDA executions.

## How the Code Works

### `benchmark.py`
The `benchmark.py` script is used to benchmark the training of the chatbot model on different devices (CPU, GPU). The key steps include:
1. **Load Training Parameters**: Loads hyperparameters from a `training_params.json` file.
2. **Preprocess Data**: Calls the `preprocess_data` function to convert training conversations into tensors suitable for model training.
3. **Run Benchmark**: Uses the `train_model` function to train the chatbot model and measure the time taken for both CPU and GPU. This helps determine the performance gain offered by GPU over CPU.

### `main.py`
This script is used for both training and testing the chatbot model. The workflow includes:
1. **Load Data and Parameters**: Training data and parameters are loaded from JSON files.
2. **Preprocess Data**: Converts conversation data into tensors, and the vocabulary is saved for future use.
3. **Train the Model**: Calls the `train_model` function to train the chatbot model with the preprocessed data and optimized hyperparameters.
4. **Inference**: After training, a few example inputs are tested with the trained model to generate responses.

### `train.py`
The `train.py` script contains the implementation of the training process using simulated quantum annealing. The key functions are:
- **`create_model`**: Creates an instance of the `SimpleChatbotModel` with specified input, hidden, and output dimensions.
- **`train_model`**: Implements the quantum-inspired optimization process, using multiprocessing to evaluate candidate hyperparameters and select the best model configuration.
- **`evaluate_candidate`**: Evaluates a candidate model configuration by training the model and calculating the average loss over a set number of epochs.

## Running the Code

### Prerequisites
- Python 3.7+
- PyTorch
- NumPy
- tqdm

### Running Benchmarks
To benchmark the training on CPU or GPU, run:
```sh
python benchmark.py
```
Follow the prompts to choose the device for benchmarking.

### Training and Testing the Model
To train the chatbot model and perform inference, run:
```sh
python main.py
```
This will train the model and test it with a set of predefined inputs.

## Quantum Annealing Considerations
- **Multiprocessing**: The multiprocessing approach for CPU allows the evaluation of multiple candidate solutions in parallel, reducing the overall training time.
- **Early Termination**: The early termination mechanism ensures that the optimization does not run for an unnecessarily long duration once convergence is detected, saving computational resources.
- **Dynamic Step Size**: The adaptation of the step size helps maintain an appropriate balance between exploration of new areas of the solution space and exploitation of known good solutions.

## Conclusion
The simulated quantum annealing approach used in this project provides an effective way to optimize the chatbot model's hyperparameters, leveraging both CPU and GPU resources to improve the training process. The benchmarking script allows users to compare the performance of training on different devices, showcasing the efficiency gains provided by GPU acceleration, while emphasizing the advantage of CPU multiprocessing for quantum annealing tasks.

## Author
- Virgil Vaduva

## License
This project is licensed under the MIT License.
