# src/architectures/efficientformer_light.py

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model, Input
import numpy as np

# === Bloc EfficientFormer simplifié ===
def efficientformer_block(inputs, filters, expansion_factor=3):
    # 1. Expansion
    x = layers.Conv2D(filters * expansion_factor, (1, 1), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 2. Depthwise conv
    x = layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # 3. Attention simplifiée
    attn = layers.GlobalAveragePooling2D()(x)
    attn = layers.Dense(filters * expansion_factor, activation='sigmoid')(attn)
    attn = layers.Reshape((1, 1, filters * expansion_factor))(attn)
    x = layers.Multiply()([x, attn])

    # 4. Réduction
    x = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # 5. Skip connection
    if inputs.shape[-1] == filters:
        x = layers.Add()([inputs, x])

    return x

# === Cosine Annealing Learning Rate Schedule ===
def cosine_annealing(epoch, initial_lr=0.001, epochs_per_cycle=50):
    return initial_lr / 2 * (np.cos(np.pi * (epoch % epochs_per_cycle) / epochs_per_cycle) + 1)

# === Modèle EfficientFormer Light ===
def build_efficientformer_model(
    input_shape=(19, 19, 31),
    filters=32,
    blocks=5,
    expansion_factor=3,
    reg_strength=1e-4,
    dropout_rate=0.3
):
    inputs = Input(shape=input_shape, name='board')

    x = layers.Conv2D(filters, (3, 3), padding='same', use_bias=False, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    for _ in range(blocks):
        x = efficientformer_block(x, filters, expansion_factor=expansion_factor)

    # Policy head
    policy = layers.Conv2D(64, (1, 1), use_bias=False, activation='relu',
                           kernel_regularizer=regularizers.l2(reg_strength))(x)
    policy = layers.BatchNormalization()(policy)
    policy = layers.Conv2D(1, (1, 1))(policy)
    policy = layers.Flatten()(policy)
    policy = layers.Activation('softmax', name='policy')(policy)

    # Value head
    value = layers.GlobalAveragePooling2D()(x)
    value = layers.Dense(40, activation='relu')(value)
    value = layers.Dropout(dropout_rate)(value)
    value = layers.Dense(1, activation='sigmoid', name='value')(value)

    model = Model(inputs=inputs, outputs=[policy, value])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
        metrics={'policy': 'categorical_accuracy', 'value': ['mse']}
    )

    return model
