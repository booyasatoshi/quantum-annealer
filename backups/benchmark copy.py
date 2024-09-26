# benchmark.py
# Version 1.2 - Enhanced error handling, GPU check, and progress reporting

import time
import torch
from train import train_model
import json
from data_preprocessing import preprocess_data

def load_training_params(file_path='training_params.json'):
    try:
        with open(file_path, 'r') as f:
            params = json.load(f)
        return params
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: {file_path} is not a valid JSON file.")
        return None

def run_benchmark(device_type, params):
    try:
        with open('training_data.json', 'r') as f:
            conversations = json.load(f)['conversations']
    except FileNotFoundError:
        print("Error: training_data.json not found.")
        return None
    except json.JSONDecodeError:
        print("Error: training_data.json is not a valid JSON file.")
        return None

    device = torch.device(device_type)
    
    input_tensor, output_tensor, vocab = preprocess_data(conversations)
    vocab_size = len(vocab)

    print(f"Starting {device_type.upper()} benchmark...")
    start_time = time.time()
    train_model(data=input_tensor.to(device), labels=output_tensor.to(device), vocab_size=vocab_size, 
                n_iterations=params['n_iterations'], step_size=params['step_size'], temp=params['temp'], 
                num_epochs=params['num_epochs'], input_dim_bounds=params['input_dim_bounds'], 
                hidden_dim_bounds=params['hidden_dim_bounds'], learning_rate_bounds=params['learning_rate_bounds'], 
                save_path=f'chatbot_model_{device_type}.pth', device=device)
    end_time = time.time()
    
    return end_time - start_time

def main():
    params = load_training_params()
    if params is None:
        return

    print("Choose the device to run the benchmark test:")
    print("1. CPU")
    if torch.cuda.is_available():
        print("2. GPU")
        print("3. Both (Comparison)")
        max_choice = 3
    else:
        print("GPU is not available on this system.")
        max_choice = 1

    while True:
        choice = input(f"Enter your choice (1-{max_choice}): ")
        if choice.isdigit() and 1 <= int(choice) <= max_choice:
            choice = int(choice)
            break
        print("Invalid choice. Please try again.")

    if choice == 1 or choice == 3:
        cpu_time = run_benchmark('cpu', params)
        if cpu_time:
            print(f"CPU benchmark completed in {cpu_time:.2f} seconds.")

    if (choice == 2 or choice == 3) and torch.cuda.is_available():
        gpu_time = run_benchmark('cuda', params)
        if gpu_time:
            print(f"GPU benchmark completed in {gpu_time:.2f} seconds.")

    if choice == 3 and cpu_time and gpu_time:
        print(f"GPU is {cpu_time / gpu_time:.2f} times faster than CPU.")

if __name__ == "__main__":
    main()