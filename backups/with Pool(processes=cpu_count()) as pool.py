            with Pool(processes=cpu_count()) as pool:
                processes = []
                manager = Manager()
                return_dict = manager.dict()

                update_thread = threading.Thread(target=update_pbar)
                update_thread.start()

                for i in range(n_iter):
                    print("Current iteration:", i)
                    print("Hard stop threshold:", hard_stop_threshold)
                    print("Solution found value:", solution_found.value)

                    if early_termination.value or (i >= hard_stop_threshold and not solution_found.value):
                        print("Early termination value: ", early_termination.value)
                        print("Solution found value: ", solution_found.value)
                        print("Early termination or hard stop triggered. Stopping optimization.")
                        break

                    candidate = curr + step_size * np.random.randn(bounds.shape[0])
                    candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
                    
                    process = pool.apply_async(evaluate_process, (candidate, i, return_dict))
                    processes.append(process)

                for i, process in enumerate(processes):
                    try:
                        process.wait(timeout=60)  # Wait for each process with a timeout
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
                            
                            # Hard stop check
                            if i >= hard_stop_threshold and not solution_found.value:
                                print("\nHard stop: 25% of candidates processed without finding a solution.")
                                early_termination.value = True
                                break
                    except concurrent.futures.TimeoutError:
                        print(f"Process {i} timed out")
                    
                    if early_termination.value:
                        break

                # Collect remaining results before terminating
                if early_termination.value:
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