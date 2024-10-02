# inference.py - Virgil Vaduva
# Version 1.10 - Added CUDA support

import torch
import json
from model import SimpleChatbotModel

def top_k_sampling(logits, k=5):
    values, indices = torch.topk(logits, k)
    probs = torch.softmax(values, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

def generate_response(model, input_sentence, vocab, device):
    model.eval()
    model.reset_state()  # Reset state at the beginning of each response generation
    inverse_vocab = {v: k for k, v in vocab.items()}
    input_tensor = torch.tensor([vocab.get(word, vocab["<pad>"]) for word in input_sentence.lower().split()]).unsqueeze(0).to(device)
    input_tensor = torch.cat([torch.tensor([[vocab["<sos>"]]]).to(device), input_tensor, torch.tensor([[vocab["<eos>"]]]).to(device)], dim=1)
    
    max_length = 50
    
    with torch.no_grad():
        hidden = None
        output, hidden = model(input_tensor, hidden)
        
        response_tokens = []
        for i in range(max_length):
            next_token = top_k_sampling(output[0, -1, :]).item()
            response_tokens.append(next_token)
            
            if next_token == vocab["<eos>"]:
                break
            
            next_input = torch.tensor([[next_token]]).to(device)
            output, hidden = model(next_input, hidden)
        
        print(f"Response tokens: {response_tokens}")
        
        response = ' '.join([inverse_vocab.get(token, "<unk>") for token in response_tokens if token not in [vocab["<pad>"], vocab["<sos>"], vocab["<eos>"]]])
        print(f"Response: {response}")
    return response

def load_model_and_vocab(save_path, vocab_path='vocab.json', device=torch.device("cpu")):
    checkpoint = torch.load(save_path, map_location=device)
    
    if 'best_params' in checkpoint:
        best_params = checkpoint['best_params']
    else:
        print("Warning: 'best_params' not found in checkpoint. Loading from best_params.json.")
        with open('best_params.json', 'r') as f:
            best_params = json.load(f)
    
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    model = SimpleChatbotModel(
        input_dim=int(best_params['input_dim']),
        hidden_dim=int(best_params['hidden_dim']),
        output_dim=int(best_params['vocab_size']),
        vocab_size=int(best_params['vocab_size'])
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, vocab, best_params

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, vocab, best_params = load_model_and_vocab('chatbot_model.pth', device=device)
    
    print("Model parameters:")
    for key, value in best_params.items():
        print(f"{key}: {value}")
    
    input_sentence = "Hello"
    response = generate_response(model, input_sentence, vocab, device)
    print(f"Bot: {response}")