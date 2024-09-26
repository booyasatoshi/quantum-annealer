import torch
from inference import load_model_and_vocab, generate_response

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = 'chatbot_model.pth'
    vocab_path = 'vocab.json'
    try:
        model, vocab, _ = load_model_and_vocab(model_path, vocab_path, device)
        print("Model and vocabulary loaded successfully.")
    except Exception as e:
        print(f"Error loading model and vocabulary: {e}")
        return

    print("Interactive testing mode. Type 'quit' or 'exit' to end the session.")

    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("Ending session. Goodbye!")
            break

        try:
            # Reset model's hidden state for each new input
            model.reset_state()  # You may need to add this method to your model class
            response = generate_response(model, user_input, vocab, device)
            print(f"Bot: {response}")
        except Exception as e:
            print(f"Error generating response: {e}")

if __name__ == "__main__":
    main()