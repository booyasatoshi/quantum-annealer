# train.py
# Version 2.2 - Ensured global termination and refactored loop to respect early termination

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import json
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, set_start_method, Manager
from model import SimpleChatbotModel
from data_preprocessing import preprocess_data
import concurrent.futures
import threading
from collections import deque

# Set the start method to 'spawn' to avoid CUDA re-initialization issues
set_start_method('spawn', force=True)

def create_model(input_dim, hidden_dim, output_dim, vocab_size, device):
    model = SimpleChatbotModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, vocab_size=vocab_size).to(device)
    return model

def evaluate_candidate(args):
    candidate, data, labels, vocab_size, progress, index, device = args
    try:
        print(f"Evaluating candidate: {candidate}")
        print(f"Data shape: {data.shape}, Labels shape: {labels.shape}, Vocab size: {vocab_size}")

        if data is None or labels are None:
            print(f"Error: Data or labels are None for candidate {candidate}")
            return float('inf')
        if data.size(0) == 0 or labels.size(0) == 0:
            print(f"Error: Data or labels are empty for candidate {candidate}")
            return float('inf')
        if vocab_size == 0:
            print(f"Error: Vocab size is zero for candidate {candidate}")
            return float('inf')

        input_dim = int(candidate[0])
        hidden_dim = int(candidate[1])
        model = create_model(input_dim, hidden_dim, vocab_size, vocab_size, device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=candidate[2])
        
        model.train()
        total_loss = 0
        num_epochs = 10  # Reduced for testing
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs, _ = model(data.to(device), None)
            if outputs is None:
                print(f"Error: Outputs are None for candidate {candidate}")
                return float('inf')
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.to(device).view(-1))
            if not torch.isfinite(loss):
                print(f"Non-finite loss encountered: {loss.item()}")
                return float('inf')
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if progress is not None:
                progress[index] += 1  # Update progress for this subprocess
        avg_loss = total_loss / num_epochs
        return avg_loss
    except Exception as e:
        print(f"Error evaluating candidate {candidate}: {str(e)}")
        import traceback
        traceback.print_exc()
        return float('inf')

def train_model(data, labels, vocab_size, n_iterations, step_size, temp, num_epochs, input_dim_bounds, hidden_dim_bounds, learning_rate_bounds, save_path, device):
    bounds = np.array([input_dim_bounds, hidden_dim_bounds, learning_rate_bounds])

    def objective_function(candidate_params, data, labels, progress, index):
        return evaluate_candidate((candidate_params, data, labels, vocab_size, progress, index, device))

    def quantum_inspired_optimization(objective, bounds, n_iter, initial_step_size, temp, data, labels):
        best = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * np.random.rand(bounds.shape[0])
        best_eval = objective(best, data, labels, None, None)
        if not torch.isfinite(torch.tensor(best_eval)):
            best_eval = float('inf')
        curr, curr_eval = best, best_eval
        scores = [best_eval]

        print(f"Initial best score: {best_eval:.4f}")

        manager = Manager()
        progress = manager.list([0] * n_iter)  # Shared progress list

        worse_accepted = 0
        significant_jumps = 0

        # Adaptive step size parameters
        step_size = initial_step_size
        step_size_min = initial_step_size * 0.01
        step_size_max = initial_step_size * 2
        adaptation_factor = 1.1

        # Dynamic early termination parameters
        min_iterations = 20
        stability_window = deque(maxlen=10)
        improvement_threshold = 0.01

        if device.type == 'cuda':
            # Sequential processing for GPU
            pbar = tqdm(total=n_iter, desc="Overall Progress", position=0, leave=True)
            early_termination = False
            for i in range(n_iter):
                if early_termination:
                    break

                try:
                    candidate = curr + step_size * np.random.randn(bounds.shape[0])
                    candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])

                    print(f"Starting evaluation of candidate {i+1}/{n_iter}")
                    candidate_eval = objective(candidate, data, labels, progress, i)
                    print(f"Completed evaluation of candidate {i+1}/{n_iter}")

                    if candidate_eval < best_eval:
                        best, best_eval = candidate, candidate_eval
                    diff = candidate_eval - curr_eval
                    t = temp / float(i + 1)
                    metropolis = np.exp(-diff / t)

                    print(f"Current temperature: {t:.6f}")
                    print(f"Current best solution: {best}")
                    print(f"Current best energy: {best_eval:.6f}")

                    if diff < 0 or random.random() < metropolis:
                        if diff > 0:
                            worse_accepted += 1
                            print(f"Accepting worse solution. Acceptance probability: {metropolis:.6f}")
                        if abs(diff) > 0.1 * curr_eval:
                            significant_jumps += 1
                            print("Significant jump in solution space detected!")
                        curr, curr_eval = candidate, candidate_eval
                    scores.append(curr_eval)

                    print(f"Exploration vs Exploitation: {worse_accepted/(i+1):.2f}")
                    print(f"Improvement rate: {(scores[0] - best_eval) / (i+1):.6f}")

                    # Update main progress bar
                    pbar.update(1)
                    pbar.refresh()

                    # Dynamic early termination check
                    stability_window.append(best_eval)
                    if i >= min_iterations:
                        avg_best = np.mean(stability_window)
                        if len(stability_window) == stability_window.maxlen:
                            relative_improvement = (avg_best - best_eval) / avg_best
                            if relative_improvement < improvement_threshold:
                                print("\nEarly termination: Satisfactory solution reached or convergence criterion met.")
                                early_termination = True

                except Exception as e:
                    print(f"Error in optimization loop: {e}")
                    continue

            if early_termination:
                pbar.n = pbar.total
                pbar.refresh()
            pbar.set_description("Optimization Completed")
            pbar.close()
        else:
            # Multi-threaded processing for CPU
            pbar = tqdm(total=n_iter, desc="Overall Progress", position=0, leave=True)
            early_termination = False
            
            def evaluate_thread(candidate, i):
                nonlocal best, best_eval, curr, curr_eval, worse_accepted, significant_jumps, step_size
                try:
                    candidate_eval = objective(candidate, data, labels, progress, i)
                    
                    with threading.Lock():
                        if candidate_eval < best_eval:
                            best, best_eval = candidate, candidate_eval
                            step_size = min(step_size * adaptation_factor, step_size_max)
                        else:
                            step_size = max(step_size / adaptation_factor, step_size_min)
                        
                        diff = candidate_eval - curr_eval
                        t = temp / float(i + 1)
                        metropolis = np.exp(-diff / t)
                        
                        if diff < 0 or random.random() < metropolis:
                            if diff > 0:
                                worse_accepted += 1
                            if abs(diff) > 0.1 * curr_eval:
                                significant_jumps += 1
                            curr, curr_eval = candidate, candidate_eval
                        scores.append(curr_eval)
                        
                        print(f"Iteration {i+1}/{n_iter}")
                        print(f"Current temperature: {t:.6f}")
                        print(f"Current best solution: {best}")
                        print(f"Current best energy: {best_eval:.6f}")
                        print(f"Current step size: {step_size:.6f}")
                        print(f"Exploration vs Exploitation: {worse_accepted/(i+1):.2f}")
                        print(f"Improvement rate: {(scores[0] - best_eval) / (i+1):.6f}")
                        
                    return candidate_eval
                except Exception as e:
                    print(f"Error in thread: {e}")
                    return float('inf')
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
                for i in range(n_iter):
                    if early_termination:
                        break

                    candidate = curr + step_size * np.random.randn(bounds.shape[0])
                    candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
                    executor.submit(evaluate_thread, candidate, i)
                    
                    # Update main progress bar
                    pbar.update(1)
                    pbar.refresh()
                    
                    # Dynamic early termination check
                    stability_window.append(best_eval)
                    if i >= min_iterations:
                        avg_best = np.mean(stability_window)
                        if len(stability_window) == stability_window.maxlen:
                            relative_improvement = (avg_best - best_eval) / avg_best
                            if relative_improvement < improvement_threshold:
                                print("\nEarly termination: Satisfactory solution reached or convergence criterion met.")
                                early_termination = True
                                break
            
            if early_termination:
                pbar.n = pbar.total
                pbar.refresh()
            pbar.set_description("Optimization Completed")
            pbar.close()
        
        print("\nSimulated Annealing Summary:")
        print(f"Initial energy: {scores[0]:.6f}")
        print(f"Final energy: {best_eval:.6f}")
        print(f"Total iterations: {len(scores)}")
        print(f"Worse solutions accepted: {worse_accepted}")
        print(f"Significant jumps: {significant_jumps}")
        print(f"Final temperature: {temp/float(len(scores)):.6f}")
        print(f"Final step size: {step_size:.6f}")

        return best, best_eval, scores, early_termination

    best_params, best_eval, scores, early_termination = quantum_inspired_optimization(objective_function, bounds, n_iterations, step_size, temp, data, labels)

    best_params_dict = dict(zip(['input_dim', 'hidden_dim', 'learning_rate'], best_params))
    best_params_dict['vocab_size'] = vocab_size
    best_params_dict['output_dim'] = vocab_size

    model = create_model(int(best_params_dict['input_dim']), int(best_params_dict['hidden_dim']), vocab_size, vocab_size, device)

    torch.save({
        'model_state_dict': model.state_dict(),
        'best_params': best_params_dict
    }, save_path)

    print(f"\nTraining summary:")
    if early_termination:
        print("Optimization completed early due to satisfactory solution or convergence.")
    print(f"Initial best score: {scores[0]:.4f}")
    print(f"Final best score: {best_eval:.4f}")
    print(f"Improvement: {scores[0] - best_eval:.4f}")
    print("Best parameters:", best_params_dict)

if __name__ == "__main__":
    print("This script is not meant to be run independently. Please use benchmark.py to run experiments.")
