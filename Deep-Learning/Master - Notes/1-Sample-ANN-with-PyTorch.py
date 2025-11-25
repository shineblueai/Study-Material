import torch
import torch.nn as nn

class SimpleANN(nn.Module):
    def __init__(self, input_size=10, output_size=1, hidden_size=5, num_hidden_layers=5):
        """
        A 5-layer Artificial Neural Network with 5 neurons per hidden layer.
        
        Args:
            input_size (int): Number of input features (e.g., 10 for tabular data)
            output_size (int): Number of output neurons (1 for binary/regression, N for multi-class)
            hidden_size (int): Number of neurons in each hidden layer (default: 5)
            num_hidden_layers (int): Number of hidden layers (default: 5)
        """
        super(SimpleANN, self).__init__()
        
        # Input layer
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        
        # Add hidden layers (total: 5 layers with 5 neurons each)
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        # Use nn.Sequential to chain layers
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Example: Binary classification (adjust output_size for your task)
model = SimpleANN(input_size=10, output_size=1)

# For multi-class (e.g., 3 classes): SimpleANN(input_size=10, output_size=3)
# For regression: SimpleANN(input_size=10, output_size=1) + no sigmoid

print(model)