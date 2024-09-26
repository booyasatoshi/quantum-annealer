# train.py
# Version 1.47 - Implemented Parallel Tempering with efficient multithreading for CPU optimization
# Updates:
# - Added Parallel Tempering to the quantum-inspired optimization function
# - Restored and improved multithreading for CPU optimization
# - Integrated Parallel Tempering with adaptive step size mechanism
# - Maintained existing GPU functionality

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

def train_model(data, labels, vocab_size, n_iterations, step_size, temp, num_epochs, input_dim_bounds, hidden_dim_bounds, learning_rate_bounds, save_path, device):
    bounds = np.array([input_dim_bounds, hidden_dim_bounds, learning_rate_bounds])

    def objective_function(candidate_params, data, labels, progress, index):
        return evaluate_candidate((candidate_params, data, labels, vocab_size, progress, index, device))

    def quantum_inspired_optimization(objective, bounds, n_iter, initial_step_size, temp, data, labels):
        num_replicas = 4  # Number of parallel tempering replicas
        temps = np.logspace(np.log10(temp), np.log10(temp/10), num_replicas)
        
        replicas = [bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * np.random.rand(bounds.shape[0]) for _ in range(num_replicas)]
        replica_evals = [objective(replica, data, labels, None, None) for replica in replicas]
        
        best = replicas[np.argmin(replica_evals)]
        best_eval = min(replica_evals)
        if not torch.isfinite(torch.tensor(best_eval)):
            best_eval = float('inf')
        scores = [best_eval]

        print(f"Initial best score: {best_eval:.4f}")

        manager = Manager()
        progress = manager.list([0] * n_iter * num_replicas)  # Shared progress list
        
        worse_accepted = 0
        significant_jumps = 0
        
        # Adaptive step size parameters
        step_sizes = [initial_step_size] * num_replicas
        step_size_min = initial_step_size * 0.01
        step_size_max = initial_step_size * 2
        adaptation_factor = 1.1
        
        if device.type == 'cuda':
            # Sequential processing for GPU
            pbar = tqdm(total=n_iter * num_epochs * num_replicas, desc="Overall Progress", position=0, leave=True)
            early_termination = False
            for i in range(n_iter):
                for j in range(num_replicas):
                    try:
                        candidate = replicas[j] + step_sizes[j] * np.random.randn(bounds.shape[0])
                        candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
                        
                        print(f"Starting evaluation of candidate {i+1}/{n_iter} for replica {j+1}/{num_replicas}")
                        candidate_eval = objective(candidate, data, labels, progress, i*num_replicas+j)
                        print(f"Completed evaluation of candidate {i+1}/{n_iter} for replica {j+1}/{num_replicas}")
                        
                        if candidate_eval < best_eval:
                            best, best_eval = candidate, candidate_eval
                        
                        diff = candidate_eval - replica_evals[j]
                        t = temps[j]
                        metropolis = np.exp(-diff / t)
                        
                        if diff < 0 or random.random() < metropolis:
                            replicas[j], replica_evals[j] = candidate, candidate_eval
                            if diff > 0:
                                worse_accepted += 1
                            if abs(diff) > 0.1 * replica_evals[j]:
                                significant_jumps += 1
                            
                            # Adjust step size
                            step_sizes[j] = min(step_sizes[j] * adaptation_factor, step_size_max)
                        else:
                            step_sizes[j] = max(step_sizes[j] / adaptation_factor, step_size_min)
                        
                        print(f"Replica {j+1} - Temperature: {t:.6f}")
                        print(f"Replica {j+1} - Current solution: {replicas[j]}")
                        print(f"Replica {j+1} - Current energy: {replica_evals[j]:.6f}")
                        print(f"Replica {j+1} - Step size: {step_sizes[j]:.6f}")
                    
                    except Exception as e:
                        print(f"Error in optimization loop: {e}")
                        continue
                
                # Replica exchange
                for j in range(num_replicas - 1):
                    diff = (1/temps[j] - 1/temps[j+1]) * (replica_evals[j+1] - replica_evals[j])
                    if diff > 0 or random.random() < np.exp(diff):
                        replicas[j], replicas[j+1] = replicas[j+1], replicas[j]
                        replica_evals[j], replica_evals[j+1] = replica_evals[j+1], replica_evals[j]
                
                scores.append(best_eval)
                
                print(f"Iteration {i+1}/{n_iter}")
                print(f"Current best solution: {best}")
                print(f"Current best energy: {best_eval:.6f}")
                print(f"Exploration vs Exploitation: {worse_accepted/((i+1)*num_replicas):.2f}")
                print(f"Improvement rate: {(scores[0] - best_eval) / (i+1):.6f}")
                
                # Update main progress bar
                pbar.n = sum(progress)
                pbar.refresh()

                # Check for early termination condition
                if best_eval < 0.01 or i > n_iter // 2 and all(abs(score - best_eval) < 0.001 for score in scores[-10:]):
                    print("\nEarly termination: Satisfactory solution reached or convergence criterion met.")
                    early_termination = True
                    break

            if early_termination:
                pbar.n = pbar.total
                pbar.refresh()
            pbar.set_description("Optimization Completed")
            pbar.close()
        else:
            # Multi-threaded processing for CPU
            pbar = tqdm(total=n_iter * num_epochs * num_replicas, desc="Overall Progress", position=0, leave=True)
            early_termination = False
            
            def evaluate_replica(replica, temp, step_size, i, j):
                nonlocal best, best_eval, worse_accepted, significant_jumps
                try:
                    candidate = replica + step_size * np.random.randn(bounds.shape[0])
                    candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
                    
                    candidate_eval = objective(candidate, data, labels, progress, i*num_replicas+j)
                    
                    with threading.Lock():
                        if candidate_eval < best_eval:
                            best, best_eval = candidate, candidate_eval
                        
                        diff = candidate_eval - replica_evals[j]
                        metropolis = np.exp(-diff / temp)
                        
                        if diff < 0 or random.random() < metropolis:
                            replicas[j], replica_evals[j] = candidate, candidate_eval
                            if diff > 0:
                                worse_accepted += 1
                            if abs(diff) > 0.1 * replica_evals[j]:
                                significant_jumps += 1
                            
                            # Adjust step size
                            step_sizes[j] = min(step_sizes[j] * adaptation_factor, step_size_max)
                        else:
                            step_sizes[j] = max(step_sizes[j] / adaptation_factor, step_size_min)
                    
                    return j, candidate, candidate_eval
                except Exception as e:
                    print(f"Error in thread: {e}")
                    return j, None, float('inf')
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
                for i in range(n_iter):
                    futures = [executor.submit(evaluate_replica, replicas[j], temps[j], step_sizes[j], i, j) for j in range(num_replicas)]
                    for future in concurrent.futures.as_completed(futures):
                        j, candidate, candidate_eval = future.result()
                        if candidate is not None:
                            replicas[j], replica_evals[j] = candidate, candidate_eval
                    
                    # Replica exchange
                    for j in range(num_replicas - 1):
                        diff = (1/temps[j] - 1/temps[j+1]) * (replica_evals[j+1] - replica_evals[j])
                        if diff > 0 or random.random() < np.exp(diff):
                            replicas[j], replicas[j+1] = replicas[j+1], replicas[j]
                            replica_evals[j], replica_evals[j+1] = replica_evals[j+1], replica_evals[j]
                    
                    scores.append(best_eval)
                    
                    print(f"Iteration {i+1}/{n_iter}")
                    print(f"Current best solution: {best}")
                    print(f"Current best energy: {best_eval:.6f}")
                    print(f"Exploration vs Exploitation: {worse_accepted/((i+1)*num_replicas):.2f}")
                    print(f"Improvement rate: {(scores[0] - best_eval) / (i+1):.6f}")
                    
                    # Update main progress bar
                    pbar.n = sum(progress)
                    pbar.refresh()
                    
                    # Check for early termination condition
                    if best_eval < 0.01 or i > n_iter // 2 and all(abs(score - best_eval) < 0.001 for score in scores[-10:]):
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
        print(f"Final temperatures: {temps}")
        print(f"Final step sizes: {step_sizes}")
        
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