import numpy as np


# =============== 1. ReLU (Rectified Linear Unit) ===============
def relu(x):
    """ReLU activation: f(x) = max(0, x)

    Why it's used in practice:
      - Helps with gradient vanishing in deep networks
      - Fast to compute (common in hidden layers of CNNs/RNNs)

    ⚠️ Critical note:
      For negative inputs → returns 0 (no gradient)
    """
    return np.maximum(0, x)


# =============== 2. Sigmoid ===============
def sigmoid(x):
    """Sigmoid activation: f(x) = 1/(1+e^{-x})

    Why it's used in practice:
      - Outputs values between 0 and 1 (great for binary classification)
      - Smooth gradient (easy to optimize)

    ⚠️ Critical note:
      Numerically stable with large inputs → add small epsilon
    """
    # Prevent overflow/underflow (e.g., x=1000 → e^1000 is huge!)
    return 1 / (1 + np.exp(-x)) if x > -50 else 1 / (1 + np.exp(-x))


# =============== 3. Tanh ===============
def tanh(x):
    """Tanh activation: f(x) = (e^x - e^{-x})/(e^x + e^{-x})

    Why it's used in practice:
      - Outputs between -1 and 1 (better for zero-centered data)
      - Similar gradient behavior to sigmoid but faster
    """
    return np.tanh(x)  # NumPy's built-in is optimized!


# =============== 4. Softmax ===============
def softmax(x):
    """Softmax activation: f_i = e^{x_i} / sum(e^{x_j})

    Why it's used in practice:
      - Converts raw scores into probabilities (for final output layer)
      - Must be applied to the **last** layer of a neural network

    ⚠️ Critical note:
      Numerical stability → subtract max(x) first to prevent overflow!
    """
    # Prevent overflow for large values
    exp_x = np.exp(x - np.max(x))  # Subtract max to stabilize
    return exp_x / np.sum(exp_x)


# ================== DEMO: Test all functions ==================
if __name__ == "__main__":
    # Generate test data (your sine wave project context!)
    test_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)

    print("=== RELU TEST ===")
    print(f"Input: {test_data}\nOutput: {relu(test_data)}\n")

    print("=== SIGMOID TEST ===")
    print(f"Input: {test_data}\nOutput: {sigmoid(test_data)}\n")

    print("=== TANH TEST ===")
    print(f"Input: {test_data}\nOutput: {tanh(test_data)}\n")

    print("=== SOFTMAX TEST (for 3 classes) ===")
    # Example input for softmax (3 classes)
    class_scores = np.array([2.0, 1.5, -0.5])
    print(f"Input scores: {class_scores}\nOutput probs: {softmax(class_scores)}\n")

# ================== YOUR NEXT STEP ==================
print("\n✅ Your next step: Use these in your sine wave project!")
print("Example: Add a ReLU layer after your sine wave processing")
print("  → `sine_wave_output = relu(sine_waves)`")
