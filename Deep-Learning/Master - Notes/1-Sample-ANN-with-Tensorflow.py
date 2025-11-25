import tensorflow as tf
from tensorflow.keras import layers, models

# Create a sequential model
model = models.Sequential()

# Input layer (you must specify input shape for the first layer)
# Example: input shape = (10,) for 10 features
model.add(layers.Dense(5, activation='relu', input_shape=(10,)))  # Layer 1

# Hidden layers (4 more layers with 5 neurons each)
model.add(layers.Dense(5, activation='relu'))  # Layer 2
model.add(layers.Dense(5, activation='relu'))  # Layer 3
model.add(layers.Dense(5, activation='relu'))  # Layer 4
model.add(layers.Dense(5, activation='relu'))  # Layer 5

# Output layer (adjust based on your task)
# For binary classification:
model.add(layers.Dense(1, activation='sigmoid'))

# For multi-class classification (e.g., 3 classes):
# model.add(layers.Dense(3, activation='softmax'))

# For regression:
# model.add(layers.Dense(1))  # No activation

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Change based on task
    metrics=['accuracy']
)

# Display model architecture
model.summary()