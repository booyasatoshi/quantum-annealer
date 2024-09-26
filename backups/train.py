# train.py
# Version 1.21 - Added debug print statements without changing functionality

import torch
import torch.optim as optim
import torch.nn as nn
from model import SimpleChatbotModel
import random
import numpy as np
import json
from tqdm import tqdm

def train_model(data, labels, vocab_size, n_iterations, step_size, temp, save_path, device):
    bounds = np.array([[50, 500], [50, 500], [0.0001, 0.1]])

    def objective_function(params, data, labels):
        model = SimpleChatbotModel(input_dim=int(params['input_dim']), hidden_dim=int(params['hidden_dim']), output_dim=vocab_size, vocab_size=vocab_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        model.train()
        total_loss = 0
        num_epochs = 10
        initial_loss = None
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs, _ = model(data, None)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if epoch == 0:
                initial_loss = loss.item()
        avg_loss = total_loss / num_epochs
        print(f"Training attempt - Initial loss: {initial_loss:.4f}, Final loss: {loss.item():.4f}")
        return avg_loss

    def quantum_inspired_optimization(objective, bounds, n_iter, step_size, temp, data, labels):
        best = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * np.random.rand(bounds.shape[0])
        best_eval = objective(dict(zip(['input_dim', 'hidden_dim', 'learning_rate'], best)), data, labels)
        curr, curr_eval = best, best_eval
        scores = [best_eval]
        
        print(f"Initial best score: {best_eval:.4f}")
        pbar = tqdm(total=n_iter, desc="Optimization Progress")
        for i in range(n_iter):
            candidate = curr + step_size * np.random.randn(bounds.shape[0])
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            candidate_eval = objective(dict(zip(['input_dim', 'hidden_dim', 'learning_rate'], candidate)), data, labels)
            if candidate_eval < best_eval:
                best, best_eval = candidate, candidate_eval
            diff = candidate_eval - curr_eval
            t = temp / float(i + 1)
            metropolis = np.exp(-diff / t)
            if diff < 0 or random.random() < metropolis:
                curr, curr_eval = candidate, candidate_eval
            scores.append(curr_eval)
            pbar.update(1)
            if i % 100 == 0 and i > 0:
                pbar.set_postfix({"Best Score": f"{best_eval:.4f}"})
        pbar.close()
        return best, best_eval, scores

    best_params, best_eval, scores = quantum_inspired_optimization(objective_function, bounds, n_iterations, step_size, temp, data, labels)
    
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
    
    print(f"Training summary:")
    print(f"Initial best score: {scores[0]:.4f}")
    print(f"Final best score: {best_eval:.4f}")
    print(f"Improvement: {scores[0] - best_eval:.4f}")
    print("Best parameters:", best_params_dict)

    with open('best_params.json', 'w') as f:
        json.dump(best_params_dict, f)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from data_preprocessing import preprocess_data
    conversations = [
        {"input": "Hello", "output": "Hi there!"},
        {"input": "How are you?", "output": "I'm doing well, thank you!"},
        {"input": "What is your name?", "output": "I am a chatbot."},
        {"input": "Tell me a joke", "output": "Why don't scientists trust atoms? Because they make up everything!"},
        {"input": "Goodbye", "output": "Goodbye! Have a nice day!"},
    ]

    input_tensor, output_tensor, vocab = preprocess_data(conversations)
    input_tensor = input_tensor.to(device)
    output_tensor = output_tensor.to(device)
    vocab_size = len(vocab)
    train_model(data=input_tensor, labels=output_tensor, vocab_size=vocab_size, n_iterations=100, step_size=0.1, temp=10, save_path='chatbot_model.pth', device=device)