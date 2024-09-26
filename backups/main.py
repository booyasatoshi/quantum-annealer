# main.py
# Version 1.6 - Added CUDA support

import json
import torch
from data_preprocessing import preprocess_data, save_vocab
from model import SimpleChatbotModel
from train import train_model
from inference import generate_response, load_model_and_vocab

def load_training_data(file_path='training_data.json'):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['conversations']

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load training data
conversations = load_training_data()

# Preprocess data
input_tensor, output_tensor, vocab = preprocess_data(conversations)

# Move tensors to device
input_tensor = input_tensor.to(device)
output_tensor = output_tensor.to(device)

# Save vocabulary
save_vocab(vocab)

# Calculate vocab_size
vocab_size = len(vocab)

# Train the model and save it
train_model(data=input_tensor, labels=output_tensor, vocab_size=vocab_size, n_iterations=10000, step_size=0.05, temp=5, save_path='chatbot_model.pth', device=device)

# Load the trained model
model, loaded_vocab, best_params = load_model_and_vocab('chatbot_model.pth', device=device)

# Test the trained model with some example inputs
test_inputs = [
    "Hello",
    "How does photosynthesis work?",
    "Tell me a joke",
    "What's the capital of France?",
    "How do I learn to code?"
]

print("\nTesting the trained model:")
for input_sentence in test_inputs:
    response = generate_response(model, input_sentence, loaded_vocab, device)
    print(f"Input: {input_sentence}")
    print(f"Bot: {response}\n")

print("Training and testing complete.")