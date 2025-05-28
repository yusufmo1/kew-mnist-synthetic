"""Simplified CNN model architecture for Kew-MNIST classification."""

from typing import Optional, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_kew_cnn(
    input_shape: Tuple[int, int, int] = (500, 500, 1),
    num_classes: int = 6,
    learning_rate: float = 0.001
) -> keras.Model:
    """Create Kew-MNIST CNN model.
    
    Architecture:
    - Input normalization
    - 3 Convolutional blocks with LeakyReLU activation
    - MaxPooling after each block (4x4, 4x4, 2x2)
    - 2 Dense layers with L2 regularization and dropout
    - Output layer with softmax activation
    
    Args:
        input_shape: Input image shape (height, width, channels).
        num_classes: Number of output classes.
        learning_rate: Initial learning rate for optimizer.
        
    Returns:
        Compiled Keras model.
    """
    print(f"Creating Kew-CNN model with input shape {input_shape} and {num_classes} classes")
    
    # Build model using functional API
    inputs = layers.Input(shape=input_shape, name="input_images")
    
    # Normalization
    x = layers.Rescaling(1.0/255, name="normalization")(inputs)
    
    # Data augmentation (only during training)
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomZoom(0.05),
    ], name="data_augmentation")
    x = data_augmentation(x, training=True)
    
    # Conv Block 1: 16 filters, 3x3 kernel
    x = layers.Conv2D(16, (3, 3), padding='same', kernel_initializer='he_uniform', name="conv2d_block1")(x)
    x = layers.LeakyReLU(alpha=0.1, name="activation_block1")(x)
    x = layers.MaxPooling2D((4, 4), name="maxpool_block1")(x)
    
    # Conv Block 2: 64 filters, 4x4 kernel
    x = layers.Conv2D(64, (4, 4), padding='same', kernel_initializer='he_uniform', name="conv2d_block2")(x)
    x = layers.LeakyReLU(alpha=0.1, name="activation_block2")(x)
    x = layers.MaxPooling2D((4, 4), name="maxpool_block2")(x)
    
    # Conv Block 3: 256 filters, 3x3 kernel
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform', name="conv2d_block3")(x)
    x = layers.LeakyReLU(alpha=0.1, name="activation_block3")(x)
    x = layers.MaxPooling2D((2, 2), name="maxpool_block3")(x)
    
    # Flatten for dense layers
    x = layers.Flatten(name="flatten")(x)
    
    # Dense layer 1: 128 units
    x = layers.Dense(128, kernel_initializer='he_uniform', 
                     kernel_regularizer=keras.regularizers.l2(0.0001), name="dense1")(x)
    x = layers.LeakyReLU(alpha=0.1, name="dense_activation1")(x)
    x = layers.Dropout(0.3, name="dropout1")(x)
    
    # Dense layer 2: 32 units
    x = layers.Dense(32, kernel_initializer='he_uniform',
                     kernel_regularizer=keras.regularizers.l2(0.0001), name="dense2")(x)
    x = layers.LeakyReLU(alpha=0.1, name="dense_activation2")(x)
    x = layers.Dropout(0.3, name="dropout2")(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name="predictions")(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name="KewCNN")
    
    # Create optimizer with learning rate schedule
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1350,
        decay_rate=0.88,
        staircase=True
    )
    
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Model compiled with {model.count_params():,} parameters")
    
    return model


def create_simple_cnn(
    input_shape: Tuple[int, int, int] = (500, 500, 1),
    num_classes: int = 6
) -> keras.Model:
    """Create a simpler CNN model for quick testing.
    
    Args:
        input_shape: Input image shape.
        num_classes: Number of output classes.
        
    Returns:
        Compiled Keras model.
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Normalization
        layers.Rescaling(1.0/255),
        
        # Conv layers
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name="SimpleCNN")
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model