# train.py
# Version 3.6 - Replaced multithreading with multiprocessing and implemented robust process cancellation

import torch
import time
import torch.optim as optim
import torch.nn as nn
import numpy as np
import json
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, set_start_method, Manager, Value
from model import SimpleChatbotModel
from data_preprocessing import preprocess_data
import concurrent.futures
import threading
from collections import deque
import sys
import datetime
import signal

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
        sys.stdout.flush()

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
        print(f"Creating model with input_dim: {input_dim}, hidden_dim: {hidden_dim}, vocab_size: {vocab_size}")
        model = create_model(input_dim, hidden_dim, vocab_size, vocab_size, device)
        print(f"Model created successfully for candidate: {candidate}")

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
        print(f"Candidate {candidate} average loss: {avg_loss}")
        sys.stdout.flush()
        return avg_loss
    except Exception as e:
        print(f"Error evaluating candidate {candidate}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return float('inf')


# def evaluate_candidate(args):
#     candidate, data, labels, vocab_size, progress, index, device = args
#     try:
#         print(f"Evaluating candidate: {candidate}")
#         print(f"Data shape: {data.shape}, Labels shape: {labels.shape}, Vocab size: {vocab_size}")
#         sys.stdout.flush()

#         if data is None or labels is None:
#             print(f"Error: Data or labels are None for candidate {candidate}")
#             return float('inf')
#         if data.size(0) == 0 or labels.size(0) == 0:
#             print(f"Error: Data or labels are empty for candidate {candidate}")
#             return float('inf')
#         if vocab_size == 0:
#             print(f"Error: Vocab size is zero for candidate {candidate}")
#             return float('inf')

#         input_dim = int(candidate[0])
#         hidden_dim = int(candidate[1])
#         model = create_model(input_dim, hidden_dim, vocab_size, vocab_size, device)
        
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.parameters(), lr=candidate[2])
        
#         model.train()
#         total_loss = 0
#         num_epochs = 10  # Reduced for testing
#         for epoch in range(num_epochs):
#             optimizer.zero_grad()
#             outputs, _ = model(data.to(device), None)
#             if outputs is None:
#                 print(f"Error: Outputs are None for candidate {candidate}")
#                 return float('inf')
#             loss = criterion(outputs.view(-1, outputs.size(-1)), labels.to(device).view(-1))
#             if not torch.isfinite(loss):
#                 print(f"Non-finite loss encountered: {loss.item()}")
#                 return float('inf')
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#             if progress is not None:
#                 progress[index] += 1  # Update progress for this subprocess
#         avg_loss = total_loss / num_epochs
#         print(f"Candidate {candidate} average loss: {avg_loss}")
#         return avg_loss
#     except Exception as e:
#         print(f"Error evaluating candidate {candidate}: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         sys.stdout.flush()
#         return float('inf')

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
        early_termination = Value('b', False)
        print("First instance of Early Termination set to ", early_termination.value)
        solution_found = Value('b', False)

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

        early_termination_state = None
        
        # Hard stop parameters
        hard_stop_threshold = int(0.50 * n_iter)

        if device.type == 'cuda':
            # Sequential processing for GPU
            pbar = tqdm(total=n_iter, desc="Overall Progress", position=0, leave=True)
            for i in range(n_iter):
                print("Current iteration:", i)
                print("Hard stop threshold:", hard_stop_threshold)
                print("Solution found value:", solution_found.value)

                if early_termination.value or (i >= hard_stop_threshold and not solution_found.value):
                    print("Early termination value: ", early_termination.value)
                    print("Solution found value: ", solution_found.value)
                    print("Early termination or hard stop triggered. Stopping optimization.")
                    break

                try:
                    candidate = curr + step_size * np.random.randn(bounds.shape[0])
                    candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])

                    print(f"Starting evaluation of candidate {i+1}/{n_iter}")
                    candidate_eval = objective(candidate, data, labels, progress, i)
                    print(f"Completed evaluation of candidate {i+1}/{n_iter}")

                    if candidate_eval < best_eval:
                        best, best_eval = candidate, candidate_eval
                        solution_found.value = True
                        print(f"Solution found at iteration {i}: {candidate_eval}")
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
                                early_termination.value = True
                                early_termination_state = {
                                    'iteration': i + 1,
                                    'best': best,
                                    'best_eval': best_eval,
                                    'scores': scores.copy(),
                                    'worse_accepted': worse_accepted,
                                    'significant_jumps': significant_jumps,
                                    'temperature': t,
                                    'step_size': step_size
                                }

                except Exception as e:
                    print(f"Error in optimization loop: {e}")
                    continue

            if early_termination.value or (i >= hard_stop_threshold and not solution_found.value):
                pbar.n = pbar.total
                pbar.refresh()
            pbar.set_description("Optimization Completed")
            pbar.close()
        else:
            # Multi-process processing for CPU
            pbar = tqdm(total=n_iter, desc="Overall Progress", position=0, leave=True)

            def evaluate_process(candidate, i, return_dict):
                try:
                    start_time = datetime.datetime.now()
                    print(f"Subprocess {i} started at {start_time}")
                    result = objective(candidate, data, labels, progress, i)
                    end_time = datetime.datetime.now()
                    print(f"Subprocess {i} ended at {end_time} with result {result}")
                    return_dict[i] = result
                    sys.stdout.flush()
                    
                except Exception as e:
                    print(f"Exception in subprocess {i}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()

            # def evaluate_process(candidate, i, return_dict):
            #     start_time = datetime.datetime.now()
            #     print(f"Subprocess {i} started at {start_time}")
            #     result = objective(candidate, data, labels, progress, i)
            #     end_time = datetime.datetime.now()
            #     print(f"Subprocess {i} ended at {end_time} with result {result}")
            #     return_dict[i] = result
            #     sys.stdout.flush()

            def update_pbar():
                while not early_termination.value:
                    completed = sum(p.ready() for p in processes)
                    print(f"Progress update: {completed}/{n_iter} processes completed")
                    pbar.n = completed
                    pbar.refresh()
                    sys.stdout.flush()
                    time.sleep(0.1)


            # def update_pbar():
            #     while not early_termination.value:
            #         completed = sum(p.ready() for p in processes)
            #         pbar.n = completed
            #         pbar.refresh()
            #         time.sleep(0.1)

            # with Pool(processes=cpu_count()) as pool:
            #     processes = []
            #     manager = Manager()
            #     return_dict = manager.dict()

            #     update_thread = threading.Thread(target=update_pbar)
            #     update_thread.start()

            #     for i in range(n_iter):
            #         print("Current iteration:", i)
            #         print("Hard stop threshold:", hard_stop_threshold)
            #         print("Solution found value:", solution_found.value)

            #         if early_termination.value or (i >= hard_stop_threshold and not solution_found.value):
            #             print("Early termination value: ", early_termination.value)
            #             print("Solution found value: ", solution_found.value)
            #             print("Early termination or hard stop triggered. Stopping optimization.")
            #             break

            #         candidate = curr + step_size * np.random.randn(bounds.shape[0])
            #         candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
                    
            #         process = pool.apply_async(evaluate_process, (candidate, i, return_dict))
            #         processes.append(process)

            #     for i, process in enumerate(processes):
            #         try:
            #             process.wait(timeout=60)  # Wait for each process with a timeout
            #             if i in return_dict:
            #                 candidate_eval = return_dict[i]
                            
            #                 if candidate_eval < best_eval:
            #                     best, best_eval = candidate, candidate_eval
            #                     solution_found.value = True
            #                     print(f"Solution found at iteration {i}: {candidate_eval}")
            #                     step_size = min(step_size * adaptation_factor, step_size_max)
            #                 else:
            #                     step_size = max(step_size / adaptation_factor, step_size_min)
                            
            #                 diff = candidate_eval - curr_eval
            #                 t = temp / float(i + 1)
            #                 metropolis = np.exp(-diff / t)
                            
            #                 if diff < 0 or random.random() < metropolis:
            #                     if diff > 0:
            #                         worse_accepted += 1
            #                     if abs(diff) > 0.1 * curr_eval:
            #                         significant_jumps += 1
            #                     curr, curr_eval = candidate, candidate_eval
            #                 scores.append(curr_eval)
                            
            #                 print(f"Iteration {i+1}/{n_iter}")
            #                 print(f"Current temperature: {t:.6f}")
            #                 print(f"Current best solution: {best}")
            #                 print(f"Current best energy: {best_eval:.6f}")
            #                 print(f"Current step size: {step_size:.6f}")
            #                 print(f"Exploration vs Exploitation: {worse_accepted/(i+1):.2f}")
            #                 print(f"Improvement rate: {(scores[0] - best_eval) / (i+1):.6f}")
                            
            #                 # Dynamic early termination check
            #                 stability_window.append(best_eval)
            #                 if i >= min_iterations and len(stability_window) == stability_window.maxlen:
            #                     avg_best = np.mean(stability_window)
            #                     relative_improvement = (avg_best - best_eval) / avg_best
            #                     if relative_improvement < improvement_threshold:
            #                         print("\nEarly termination: Satisfactory solution reached or convergence criterion met.")
            #                         early_termination.value = True
            #                         early_termination_state = {
            #                             'iteration': i + 1,
            #                             'best': best,
            #                             'best_eval': best_eval,
            #                             'scores': scores.copy(),
            #                             'worse_accepted': worse_accepted,
            #                             'significant_jumps': significant_jumps,
            #                             'temperature': t,
            #                             'step_size': step_size
            #                         }
            #                         break
                            
            #                 # Hard stop check
            #                 if i >= hard_stop_threshold and not solution_found.value:
            #                     print("\nHard stop: 25% of candidates processed without finding a solution.")
            #                     early_termination.value = True
            #                     break
            #         except concurrent.futures.TimeoutError:
            #             print(f"Process {i} timed out")
                    
            #         if early_termination.value:
            #             break

            #     # Collect remaining results before terminating
            #     if early_termination.value:
            #         for process in processes:
            #             process.wait(timeout=60)  # Give some time for remaining processes to finish

            #         for i, process in enumerate(processes):
            #             if i in return_dict:
            #                 candidate_eval = return_dict[i]
            #                 if candidate_eval < best_eval:
            #                     best, best_eval = candidate, candidate_eval
            #                     step_size = min(step_size * adaptation_factor, step_size_max)
            #                 else:
            #                     step_size = max(step_size / adaptation_factor, step_size_min)
                            
            #                 diff = candidate_eval - curr_eval
            #                 t = temp / float(i + 1)
            #                 metropolis = np.exp(-diff / t)
                            
            #                 if diff < 0 or random.random() < metropolis:
            #                     if diff > 0:
            #                         worse_accepted += 1
            #                     if abs(diff) > 0.1 * curr_eval:
            #                         significant_jumps += 1
            #                     curr, curr_eval = candidate, candidate_eval
            #                 scores.append(curr_eval)
                            
            #                 # Dynamic early termination check
            #                 stability_window.append(best_eval)
            #                 if i >= min_iterations and len(stability_window) == stability_window.maxlen:
            #                     avg_best = np.mean(stability_window)
            #                     relative_improvement = (avg_best - best_eval) / avg_best
            #                     if relative_improvement < improvement_threshold:
            #                         early_termination_state = {
            #                             'iteration': i + 1,
            #                             'best': best,
            #                             'best_eval': best_eval,
            #                             'scores': scores.copy(),
            #                             'worse_accepted': worse_accepted,
            #                             'significant_jumps': significant_jumps,
            #                             'temperature': t,
            #                             'step_size': step_size
            #                         }
            #                         break

            #             if not process.ready():
            #                 process.terminate()

            #     early_termination.value = True
            #     update_thread.join()

            with Pool(processes=cpu_count()) as pool:
                processes = []
                manager = Manager()
                return_dict = manager.dict()

                update_thread = threading.Thread(target=update_pbar)
                update_thread.start()

                # Initialize subprocesses
                for i in range(n_iter):
                    candidate = curr + step_size * np.random.randn(bounds.shape[0])
                    candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])

                    process = pool.apply_async(evaluate_process, (candidate, i, return_dict))
                    processes.append(process)

                    print("Current iteration:", i)
                    print("Hard stop threshold:", hard_stop_threshold)
                    print("Solution found value:", solution_found.value)

                # Wait for all subprocesses to complete
                for process in processes:
                    process.wait()

                # Print contents of return_dict after all processes have run
                print("Contents of return_dict after all processes have run:", dict(return_dict))

                # Monitor and collect results from subprocesses
                for i, process in enumerate(processes):
                    try:
                        if i in return_dict:
                            candidate_eval = return_dict[i]
                            if candidate_eval < best_eval:
                                best, best_eval = candidate, candidate_eval
                                solution_found.value = True
                                print(f"Solution found at iteration {i}: {candidate_eval}")
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

                            # Dynamic early termination check
                            stability_window.append(best_eval)
                            if i >= min_iterations and len(stability_window) == stability_window.maxlen:
                                avg_best = np.mean(stability_window)
                                relative_improvement = (avg_best - best_eval) / avg_best
                                if relative_improvement < improvement_threshold:
                                    print("\nEarly termination: Satisfactory solution reached or convergence criterion met.")
                                    early_termination.value = True
                                    early_termination_state = {
                                        'iteration': i + 1,
                                        'best': best,
                                        'best_eval': best_eval,
                                        'scores': scores.copy(),
                                        'worse_accepted': worse_accepted,
                                        'significant_jumps': significant_jumps,
                                        'temperature': t,
                                        'step_size': step_size
                                    }
                                    break

                    except concurrent.futures.TimeoutError:
                        print(f"Process {i} timed out")

                # Print contents of return_dict before terminating
                print("Contents of return_dict before termination:", dict(return_dict))

                # Ensure all results are collected before terminating
                if early_termination.value:
                    print("Early termination triggered. Allowing subprocesses to finish.")
                    for process in processes:
                        process.wait(timeout=60)  # Give some time for remaining processes to finish

                    for i, process in enumerate(processes):
                        if i in return_dict:
                            candidate_eval = return_dict[i]
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

                            # Dynamic early termination check
                            stability_window.append(best_eval)
                            if i >= min_iterations and len(stability_window) == stability_window.maxlen:
                                avg_best = np.mean(stability_window)
                                relative_improvement = (avg_best - best_eval) / avg_best
                                if relative_improvement < improvement_threshold:
                                    early_termination_state = {
                                        'iteration': i + 1,
                                        'best': best,
                                        'best_eval': best_eval,
                                        'scores': scores.copy(),
                                        'worse_accepted': worse_accepted,
                                        'significant_jumps': significant_jumps,
                                        'temperature': t,
                                        'step_size': step_size
                                    }
                                    break

                        if not process.ready():
                            process.terminate()

                early_termination.value = True
                update_thread.join()



            if early_termination.value or (i >= hard_stop_threshold and not solution_found.value):
                pbar.n = pbar.total
                pbar.refresh()
            pbar.set_description("Optimization Completed")
            pbar.close()

            final_state = early_termination_state if early_termination.value else {
                'iteration': n_iter,
                'best': best,
                'best_eval': best_eval,
                'scores': scores,
                'worse_accepted': worse_accepted,
                'significant_jumps': significant_jumps,
                'temperature': temp / float(len(scores)),
                'step_size': step_size
            }

            print("\nSimulated Annealing Summary:")
            print(f"Initial energy: {scores[0]:.6f}")
            if final_state:
                print(f"Final energy: {final_state['best_eval']:.6f}")
                print(f"Total iterations: {final_state['iteration']}")
                print(f"Worse solutions accepted: {final_state['worse_accepted']}")
                print(f"Significant jumps: {final_state['significant_jumps']}")
                print(f"Final temperature: {final_state['temperature']:.6f}")
                print(f"Final step size: {final_state['step_size']:.6f}")
            else:
                print("Optimization terminated early without producing a final state.")

            return best, best_eval, scores, early_termination.value, final_state


    best_params, best_eval, scores, early_termination, final_state = quantum_inspired_optimization(objective_function, bounds, n_iterations, step_size, temp, data, labels)

    if best_params is not None and best_eval is not None and scores:
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
        print(f"Total improvement: {scores[0] - best_eval:.4f}")
        if final_state and isinstance(final_state, dict) and 'iteration' in final_state:
            print(f"Total iterations: {final_state['iteration']}")
        else:
            print("Total iterations: Unknown")
        print("Best parameters:", best_params_dict)

        return best_params_dict
    else:
        print("Optimization did not produce valid results.")
        return None

# if __name__ == '__main__':
#     set_start_method('spawn')

if __name__ == "__main__":
    print("This script is not meant to be run independently. Please use benchmark.py to run experiments.")