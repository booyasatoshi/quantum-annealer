import numpy as np

def simulated_annealing(objective, bounds, n_iterations, step_size, temp):
    # Generate an initial point
    best = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    best_eval = objective(best)
    curr, curr_eval = best, best_eval
    
    scores = [best_eval]
    
    for i in range(n_iterations):
        # Take a step
        candidate = curr + np.random.randn(len(bounds)) * step_size
        
        # Evaluate candidate point
        candidate_eval = objective(candidate)
        
        # Check for new best solution
        if candidate_eval < best_eval:
            best, best_eval = candidate, candidate_eval
            scores.append(best_eval)
            print(f"New best solution: {best}, score: {best_eval}")
        else:
            diff = candidate_eval - curr_eval
            t = temp / float(i + 1)
            metropolis = np.exp(-diff / t)
            if diff < 0 or np.random.rand() < metropolis:
                curr, curr_eval = candidate, candidate_eval
                print(f"Accepted worse solution: {curr}, score: {curr_eval}")
    
    return [best, best_eval, scores]

# Define an objective function
def objective_function(x):
    return x[0]**2.0 + x[1]**2.0

# Define the bounds of the solution space
bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])

# Run the simulated annealing algorithm
n_iterations = 1000
step_size = 0.1
temp = 10

best, best_eval, scores = simulated_annealing(objective_function, bounds, n_iterations, step_size, temp)
print(f"Best solution: {best}, score: {best_eval}")

