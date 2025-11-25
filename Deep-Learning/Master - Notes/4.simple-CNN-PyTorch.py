import torch
import torch.nn as nn
import torch.optim as optim

class EnterpriseLSTM(nn.Module):
    """
    A production-grade LSTM network for sequence modeling in enterprise environments.
    
    Designed for use cases leveraging your Kyndryl/IBM background:
    - IT log anomaly detection
    - System metrics forecasting (CPU, memory, disk)
    - Customer behavior sequence analysis
    - Preprocessing layer for Generative AI pipelines (e.g., RAG over logs)
    
    Architecture:
        Input → [Embedding (optional)] → LSTM → Dense → Output
    """
    
    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=2,
        output_size=1,
        dropout=0.3,
        use_embedding=False,
        vocab_size=None,
        embed_dim=None
    ):
        """
        Initialize the LSTM model.
        
        Args:
            input_size (int): 
                - If use_embedding=False: number of features per timestep (e.g., 5 for system metrics)
                - If use_embedding=True: ignored (use vocab_size instead)
            hidden_size (int): Number of hidden units in LSTM (default: 128)
            num_layers (int): Number of stacked LSTM layers (default: 2)
            output_size (int): Number of outputs (1 for regression/binary, N for multi-class)
            dropout (float): Dropout rate for regularization (0.2–0.5 recommended)
            use_embedding (bool): Set True for tokenized text/logs
            vocab_size (int): Size of vocabulary (required if use_embedding=True)
            embed_dim (int): Embedding dimension (required if use_embedding=True)
        """
        super(EnterpriseLSTM, self).__init__()
        
        self.use_embedding = use_embedding
        self.output_size = output_size
        
        # Optional embedding layer (for tokenized logs)
        if use_embedding:
            assert vocab_size is not None and embed_dim is not None, \
                "vocab_size and embed_dim required when use_embedding=True"
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            lstm_input_size = embed_dim
        else:
            lstm_input_size = input_size
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): 
                - Shape (batch_size, seq_len, input_size) if use_embedding=False
                - Shape (batch_size, seq_len) if use_embedding=True (token IDs)
        
        Returns:
            output (Tensor): Predictions of shape (batch_size, output_size)
        """
        # Apply embedding if needed
        if self.use_embedding:
            x = self.embedding(x)  # (batch, seq, embed_dim)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch, seq, hidden_size)
        
        # Use output from last timestep
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Final prediction
        output = self.fc(last_output)
        return output

# ================================
# EXAMPLE 1: Time-Series Forecasting (System Metrics)
# ================================
# Input: 5 features (CPU%, memory%, disk%, network%, temp) over 50 timesteps
model_ts = EnterpriseLSTM(
    input_size=5,           # 5 system metrics per timestep
    hidden_size=128,
    num_layers=2,
    output_size=1,          # Predict next CPU usage (regression)
    dropout=0.3
)

# ================================
# EXAMPLE 2: Log Anomaly Detection (Tokenized Logs)
# ================================
model_logs = EnterpriseLSTM(
    input_size=None,        # Not used
    use_embedding=True,
    vocab_size=10000,       # Unique tokens in system logs
    embed_dim=64,
    hidden_size=128,
    output_size=2,          # [0: normal, 1: anomaly]
    dropout=0.3
)

print("Time-Series Model Architecture:")
print(model_ts)