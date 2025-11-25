import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.optimizers import Adam
import numpy as np

# ================================
# CONFIGURATION
# ================================
VOCAB_SIZE = 1000    # Size of vocabulary (for text data)
EMBEDDING_DIM = 64   # Embedding dimension
RNN_UNITS = 128      # Number of RNN units (neurons in the layer)
SEQUENCE_LENGTH = 10 # Length of input sequences
OUTPUT_SIZE = VOCAB_SIZE  # For next-token prediction

# ================================
# BUILD THE RNN MODEL
# ================================
model = Sequential([
    # Embedding layer (converts integer tokens to dense vectors)
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=SEQUENCE_LENGTH),
    
    # RNN Layer (SimpleRNN = basic recurrent layer)
    SimpleRNN(units=RNN_UNITS, return_sequences=False, activation='tanh'),
    
    # Output layer (predict next token in vocabulary)
    Dense(OUTPUT_SIZE, activation='softmax')
])

# ================================
# COMPILE THE MODEL
# ================================
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ================================
# DISPLAY MODEL ARCHITECTURE
# ================================
model.summary()