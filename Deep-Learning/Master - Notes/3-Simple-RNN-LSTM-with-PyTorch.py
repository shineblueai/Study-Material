import torch
import torch.nn as nn
import torch.optim as optim

class EnterpriseLSTM(nn.Module):
    """
    A production-grade LSTM network for sequence modeling.
    
    Designed for use cases like:
    - IT log anomaly detection (leveraging your Kyndryl/IBM ops experience)
    - Customer behavior prediction
    - Time-series forecasting
    - Preprocessing for Generative AI pipelines
    
    Architecture:
        Input → Embedding (optional) → LSTM → Dense → Output
    """
    
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        num_layers=2,
        output_dim=1,
        dropout=0.3,
        use_embedding=False,
        vocab_size=None,
        embed_dim=None
    ):
        """
        Initialize the LSTM model.
        
        Args:
            input_dim (int): 
                - If use_embedding=False: number of features per timestep (e.g., 5 for CPU, memory, disk, etc.)
                - If use_embedding=True: ignored (vocab_size used instead)
            hidden_dim (int): Number of hidden units in LSTM (default: 128)
            num_layers (int): Number of stacked LSTM layers (default: 2)
            output_dim (int): Number of output classes (1 for regression/binary, N for multi-class)
            dropout (float): Dropout rate for regularization (0.2–0.5 recommended)
            use_embedding (bool): Set True for text/token inputs (e.g., system logs)
            vocab_size (int): Size of vocabulary (required if use_embedding=True)
            embed_dim (int): Embedding dimension (required if use_embedding=True)
        """
        super(EnterpriseLSTM, self).__init__()
        
        self.use_embedding = use_embedding
        self.output_dim = output_dim
        
        # Optional embedding layer (for tokenized logs/text)
        if use_embedding:
            assert vocab_size is not None and embed_dim is not None, \
                "vocab_size and embed_dim required when use_embedding=True"
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            lstm_input_dim = embed_dim
        else:
            lstm_input_dim = input_dim
        
        # LSTM layer with dropout
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_dim)
        
        # Activation (applied in forward based on task)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
                        or (batch_size, seq_len) if use_embedding=True
        
        Returns:
            output (Tensor): Predictions of shape (batch_size, output_dim)
        """
        # Apply embedding if needed
        if self.use_embedding:
            x = self.embedding(x)  # (batch, seq, embed_dim)
        
        # Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use output from last timestep
        last_output = lstm_out[:, -1, :]  # (batch, hidden_dim)
        
        # Dense layers
        x = self.relu(self.fc1(last_output))
        x = self.dropout(x)
        output = self.fc2(x)
        
        # For binary classification, apply sigmoid externally (use BCEWithLogitsLoss)
        return output

# ================================
# EXAMPLE 1: Time-Series (e.g., System Metrics)
# ================================
# Input: 5 features (CPU, memory, disk, network, temp) over 50 timesteps
model_ts = EnterpriseLSTM(
    input_dim=5,           # 5 system metrics
    hidden_dim=128,
    num_layers=2,
    output_dim=1,          # Predict next CPU usage (regression)
    dropout=0.3
)

# ================================
# EXAMPLE 2: Log Anomaly Detection (Tokenized Logs)
# ================================
model_logs = EnterpriseLSTM(
    input_dim=None,        # Not used
    use_embedding=True,
    vocab_size=10000,      # Unique log tokens
    embed_dim=64,
    hidden_dim=128,
    output_dim=2,          # [normal, anomaly]
    dropout=0.3
)

print("Time-Series Model:")
print(model_ts)