# model.py - Virgil Vaduva
# Simulated quantum annealing staging
# Version 1.4 - Enhanced for potential multi-GPU support

import torch
import torch.nn as nn

class SimpleChatbotModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, vocab_size):
        super(SimpleChatbotModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        encoder_output, hidden = self.encoder(embedded, hidden)
        attention_weights = torch.softmax(self.attention(encoder_output), dim=1)
        context_vector = attention_weights * encoder_output
        decoder_output, hidden = self.decoder(context_vector, hidden)
        output = self.dropout(decoder_output)
        output = torch.relu(self.fc1(output))
        output = self.fc2(output)
        return output, hidden

    def reset_state(self):
        # This method resets any stateful parts of your model
        if hasattr(self, 'encoder'):
            for layer in self.encoder._all_weights:
                for param in layer:
                    if 'weight' in param:
                        nn.init.orthogonal_(self.encoder.__getattr__(param))
                    elif 'bias' in param:
                        nn.init.constant_(self.encoder.__getattr__(param), 0)
        if hasattr(self, 'decoder'):
            for layer in self.decoder._all_weights:
                for param in layer:
                    if 'weight' in param:
                        nn.init.orthogonal_(self.decoder.__getattr__(param))
                    elif 'bias' in param:
                        nn.init.constant_(self.decoder.__getattr__(param), 0)
        
# class SimpleChatbotModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, vocab_size):
#         super(SimpleChatbotModel, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, input_dim)
#         self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
#         self.attention = nn.Linear(hidden_dim, hidden_dim)
#         self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)
    
#     def forward(self, x, hidden):
#         embedded = self.embedding(x)
#         encoder_output, hidden = self.encoder(embedded, hidden)
#         attention_weights = torch.softmax(self.attention(encoder_output), dim=1)
#         context_vector = attention_weights * encoder_output
#         decoder_output, hidden = self.decoder(context_vector, hidden)
#         output = self.fc(decoder_output)
#         return output, hidden
