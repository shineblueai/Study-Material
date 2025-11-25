import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ================================
# CONFIGURATION (Customize for your use case)
# ================================
INPUT_SHAPE = (224, 224, 3)   # Image dimensions (height, width, channels)
NUM_CLASSES = 5               # e.g., 5 categories in your dataset
CONV_FILTERS = [32, 64, 128]  # Number of filters in each conv block
DROPOUT_RATE = 0.5            # Regularization to prevent overfitting

# ================================
# BUILD THE CNN MODEL
# ================================
model = Sequential(name="Enterprise_CNN")

# Block 1: Conv → Pool
model.add(Conv2D(
    filters=CONV_FILTERS[0],
    kernel_size=(3, 3),
    activation='relu',
    input_shape=INPUT_SHAPE,
    name="conv2d_block1"
))
model.add(MaxPooling2D(pool_size=(2, 2), name="maxpool_block1"))

# Block 2: Conv → Pool
model.add(Conv2D(
    filters=CONV_FILTERS[1],
    kernel_size=(3, 3),
    activation='relu',
    name="conv2d_block2"
))
model.add(MaxPooling2D(pool_size=(2, 2), name="maxpool_block2"))

# Block 3: Conv → Pool
model.add(Conv2D(
    filters=CONV_FILTERS[2],
    kernel_size=(3, 3),
    activation='relu',
    name="conv2d_block3"
))
model.add(MaxPooling2D(pool_size=(2, 2), name="maxpool_block3"))

# Classifier Head
model.add(Flatten(name="flatten"))
model.add(Dense(128, activation='relu', name="dense_hidden"))
model.add(Dropout(DROPOUT_RATE, name="dropout"))
model.add(Dense(NUM_CLASSES, activation='softmax', name="output_layer"))

# ================================
# COMPILE THE MODEL
# ================================
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',  # Use 'sparse_categorical_crossentropy' for integer labels
    metrics=['accuracy']
)

# ================================
# DISPLAY ARCHITECTURE
# ================================
model.summary()