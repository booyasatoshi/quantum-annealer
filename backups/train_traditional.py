# train.py
import torch
import torch.optim as optim
import torch.nn as nn
from data_preprocessing import preprocess_data
from model import SimpleChatbotModel

def train_model(model, data, labels, n_iterations, step_size, temp, save_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_params = {name: param.data.clone() for name, param in model.named_parameters()}
    best_loss = float('inf')
    
    for i in range(n_iterations):
        model.train()
        optimizer.zero_grad()
        outputs, _ = model(data, None)
        loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_params = {name: param.data.clone() for name, param in model.named_parameters()}
            best_loss = loss.item()
            print(f"New best solution: Loss={best_loss}")
        else:
            diff = loss.item() - best_loss
            t = temp / float(i + 1)
            metropolis = torch.exp(-diff / t)
            if diff < 0 or torch.rand(1).item() < metropolis:
                print(f"Accepted worse solution: Loss={loss.item()}")
    
    # Load the best parameters found during simulated annealing
    for name, param in model.named_parameters():
        param.data.copy_(best_params[name])
    
    # Save the trained model's state dictionary
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Example usage
conversations = [
    {"input": "Hello", "output": "Hi there!"},
    {"input": "How are you?", "output": "I'm doing well, thank you!"},
    {"input": "What is your name?", "output": "I am a chatbot."},
    {"input": "Tell me a joke", "output": "Why don't scientists trust atoms? Because they make up everything!"},
    {"input": "Goodbye", "output": "Goodbye! Have a nice day!"},
]

input_tensor, output_tensor, vocab = preprocess_data(conversations)
model = SimpleChatbotModel(input_dim=128, hidden_dim=256, output_dim=len(vocab), vocab_size=len(vocab))
train_model(model, input_tensor, output_tensor, n_iterations=100, step_size=0.1, temp=10, save_path='chatbot_model.pth')

