import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

# ================================
# CONFIGURATION (Customize for your use case)
# ================================
VOCAB_SIZE = 5000      # e.g., number of unique words/tokens
EMBEDDING_DIM = 128    # Dense vector size for each token
MAX_SEQUENCE_LENGTH = 100  # Length of input sequences
LSTM_UNITS = 256       # Number of LSTM units (higher = more capacity)
DROPOUT_RATE = 0.3     # Regularization to prevent overfitting
OUTPUT_SIZE = 2        # Binary classification (e.g., anomaly/no anomaly)

# ================================
# BUILD THE LSTM MODEL
# ================================
model = Sequential([
    # 1. Embedding Layer: Converts token IDs → dense vectors
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_SEQUENCE_LENGTH,
        name="embedding_layer"
    ),
    
    # 2. LSTM Layer: Processes sequences with memory gates
    LSTM(
        units=LSTM_UNITS,
        return_sequences=False,  # Only return last timestep output
        dropout=DROPOUT_RATE,    # Dropout on input connections
        recurrent_dropout=DROPOUT_RATE,  # Dropout on recurrent connections
        name="lstm_layer"
    ),
    
    # 3. Dense Output Layer
    Dense(64, activation='relu', name="dense_hidden"),
    Dropout(DROPOUT_RATE),
    Dense(OUTPUT_SIZE, activation='softmax', name="output_layer")  # Use 'sigmoid' for binary
])

# ================================
# COMPILE THE MODEL
# ================================
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',  # For integer labels
    metrics=['accuracy']
)

# ================================
# DISPLAY ARCHITECTURE
# ================================
model.summary()