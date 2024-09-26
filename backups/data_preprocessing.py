# data_preprocessing.py
# Version 1.4 - Fix indexing issue and ensure consistent vocabulary

import itertools
from collections import Counter
import torch
import json

def preprocess_data(conversations):
    print("Starting data preprocessing...")
    input_texts = [conv["input"] for conv in conversations]
    output_texts = [conv["output"] for conv in conversations]
    
    # Tokenize the sentences
    all_texts = input_texts + output_texts
    all_words = list(itertools.chain(*[text.lower().split() for text in all_texts]))

    # Build vocabulary
    word_counter = Counter(all_words)
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}  # Start with special tokens
    for word, _ in word_counter.items():
        if word not in vocab:
            vocab[word] = len(vocab)

    print("Vocabulary created:")
    print(vocab)

    def encode_sentence(sentence, vocab):
        tokens = ["<sos>"] + sentence.lower().split() + ["<eos>"]
        return [vocab.get(token, vocab["<pad>"]) for token in tokens]

    # Encode input and output sentences
    encoded_inputs = [encode_sentence(conv["input"], vocab) for conv in conversations]
    encoded_outputs = [encode_sentence(conv["output"], vocab) for conv in conversations]

    print("Encoded inputs:", encoded_inputs)
    print("Encoded outputs:", encoded_outputs)

    # Pad sequences to the same length
    max_length = max(max(len(seq) for seq in encoded_inputs), max(len(seq) for seq in encoded_outputs))

    def pad_sequence(sequence, max_length, pad_value=0):
        return sequence + [pad_value] * (max_length - len(sequence))

    encoded_inputs = [pad_sequence(seq, max_length) for seq in encoded_inputs]
    encoded_outputs = [pad_sequence(seq, max_length) for seq in encoded_outputs]

    # Convert to tensors
    input_tensor = torch.tensor(encoded_inputs, dtype=torch.long)
    output_tensor = torch.tensor(encoded_outputs, dtype=torch.long)

    return input_tensor, output_tensor, vocab

def save_vocab(vocab, file_path='vocab.json'):
    with open(file_path, 'w') as f:
        json.dump(vocab, f)
    print(f"Vocabulary saved to {file_path}")

# Example usage
if __name__ == "__main__":
    conversations = [
        {"input": "Hello", "output": "Hi there!"},
        {"input": "How are you?", "output": "I'm doing well, thank you!"},
        {"input": "What is your name?", "output": "I am a chatbot."},
        {"input": "Tell me a joke", "output": "Why don't scientists trust atoms? Because they make up everything!"},
        {"input": "Goodbye", "output": "Goodbye! Have a nice day!"},
    ]

    input_tensor, output_tensor, vocab = preprocess_data(conversations)
    print("Input Tensor:\n", input_tensor)
    print("Output Tensor:\n", output_tensor)
    print("Vocabulary:\n", vocab)
    
    save_vocab(vocab)