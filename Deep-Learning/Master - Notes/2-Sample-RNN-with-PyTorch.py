import torch
import torch.nn as nn
import torch.optim as optim

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=1, output_size=None):
        """
        A customizable Recurrent Neural Network (RNN) for sequence modeling.
        
        Args:
            vocab_size (int): Size of the input vocabulary (e.g., 1000 unique tokens).
            embed_dim (int): Dimension of word/token embeddings (default: 64).
            hidden_dim (int): Number of features in the hidden state (default: 128).
            num_layers (int): Number of stacked RNN layers (default: 1).
            output_size (int): Size of output (default: vocab_size for next-token prediction).
        """
        super(SimpleRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_size = output_size if output_size else vocab_size
        
        # Embedding layer: maps token IDs to dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # RNN layer: processes sequences step-by-step
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,     # Input/Output format: (batch, seq, features)
            nonlinearity='tanh'   # Activation function
        )
        
        # Output layer: predicts next token or class
        self.fc = nn.Linear(hidden_dim, self.output_size)
        
    def forward(self, x, hidden=None):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length)
            hidden (Tensor): Optional initial hidden state (default: zeros)
        
        Returns:
            output (Tensor): Predictions of shape (batch_size, output_size)
            hidden (Tensor): Final hidden state for sequence
        """
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        
        # Embed input tokens
        embedded = self.embedding(x)  # (batch, seq, embed_dim)
        
        # Pass through RNN
        rnn_out, hidden = self.rnn(embedded, hidden)  # rnn_out: (batch, seq, hidden_dim)
        
        # Use output from the last timestep
        output = self.fc(rnn_out[:, -1, :])  # (batch, output_size)
        
        return output, hidden

# ================================
# EXAMPLE: Binary Classification or Next-Token Prediction
# ================================
VOCAB_SIZE = 1000
model = SimpleRNN(vocab_size=VOCAB_SIZE, hidden_dim=128, num_layers=2)

# For regression or binary classification, set output_size explicitly:
# model = SimpleRNN(vocab_size=50, output_size=1)  # e.g., time-series forecasting

print(model)