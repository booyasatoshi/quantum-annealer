# train.py
# Version 1.42 - Updated to use the device passed from benchmark.py

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

# Set the start method to 'spawn' to avoid CUDA re-initialization issues
set_start_method('spawn', force=True)

def evaluate_candidate(args):
    candidate, data, labels, vocab_size, progress, index, device = args
    try:
        print(f"Evaluating candidate: {candidate}")
        print(f"Data shape: {data.shape}, Labels shape: {labels.shape}, Vocab size: {vocab_size}")
        
        if data is None or labels is None:
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
        output_dim = vocab_size
        
        print(f"Creating model with input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}, vocab_size={vocab_size}")
        
        model = SimpleChatbotModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, vocab_size=vocab_size).to(device)
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

def run_multiprocessing(data, labels, vocab_size, device):
    bounds = np.array([[50, 500], [50, 500], [0.0001, 0.1]])
    candidates = [bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * np.random.rand(bounds.shape[0]) for _ in range(10)]
    
    manager = Manager()
    progress = manager.list([0] * len(candidates))
    
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(evaluate_candidate, [(c, data, labels, vocab_size, progress, i, device) for i, c in enumerate(candidates)]), total=len(candidates), desc="Evaluating candidates"))
    
    print("Results:", results)

def train_model(data, labels, vocab_size, n_iterations, step_size, temp, num_epochs, input_dim_bounds, hidden_dim_bounds, learning_rate_bounds, save_path, device):
    bounds = np.array([input_dim_bounds, hidden_dim_bounds, learning_rate_bounds])

    def objective_function(candidate_params, data, labels, progress, index):
        return evaluate_candidate((candidate_params, data, labels, vocab_size, progress, index, device))

    def quantum_inspired_optimization(objective, bounds, n_iter, step_size, temp, data, labels):
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
        
        with Pool(processes=cpu_count()) as pool:
            pbar = tqdm(total=n_iter * num_epochs, desc="Overall Progress", position=0, leave=True)
            early_termination = False
            for i in range(n_iter):
                try:
                    candidate = curr + step_size * np.random.randn(bounds.shape[0])
                    candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
                    
                    candidate_params = [candidate]
                    args = [(candidate, data, labels, vocab_size, progress, i, device) for candidate in candidate_params]
                    
                    print(f"Starting evaluation of candidate {i+1}/{n_iter}")
                    candidate_eval = pool.map(evaluate_candidate, args)[0]
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
                    overall_progress = sum(progress)
                    pbar.n = overall_progress
                    pbar.refresh()

                    # Check for early termination condition (example)
                    if best_eval < 0.01 or i > n_iter // 2 and all(abs(score - best_eval) < 0.001 for score in scores[-10:]):
                        print("\nEarly termination: Satisfactory solution reached or convergence criterion met.")
                        early_termination = True
                        break

                except Exception as e:
                    print(f"Error in optimization loop: {e}")
                    continue

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
        
        return best, best_eval, scores, early_termination

    best_params, best_eval, scores, early_termination = quantum_inspired_optimization(objective_function, bounds, n_iterations, step_size, temp, data, labels)

    best_params_dict = dict(zip(['input_dim', 'hidden_dim', 'learning_rate'], best_params))
    best_params_dict['vocab_size'] = vocab_size
    best_params_dict['output_dim'] = vocab_size

    model = SimpleChatbotModel(input_dim=int(best_params_dict['input_dim']),
                               hidden_dim=int(best_params_dict['hidden_dim']),
                               output_dim=vocab_size,
                               vocab_size=vocab_size).to(device)

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