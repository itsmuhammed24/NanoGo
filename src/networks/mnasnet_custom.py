# src/architectures/mnasnet_custom.py

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model, Input
import numpy as np

# === SE Block ===
def SE_Block(inputs, ratio=16):
    filters = inputs.shape[-1]
    se = layers.GlobalAveragePooling2D()(inputs)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    return layers.Multiply()([inputs, se])

# === Channel Attention Block ===
def channel_attention(inputs, ratio=8):
    filters = inputs.shape[-1]
    x = layers.GlobalAveragePooling2D()(inputs)
    x = layers.Reshape((1, 1, filters))(x)
    x = layers.Dense(filters // ratio, activation='relu')(x)
    x = layers.Dense(filters, activation='sigmoid')(x)
    return layers.Multiply()([inputs, x])

# === MNASNet-inspired block with dual attention ===
def mnasnet_block_with_attention(inputs, filters, kernel_size, expansion_factor):
    x = layers.Conv2D(filters * expansion_factor, (1, 1), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.DepthwiseConv2D(kernel_size, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    x = SE_Block(x)
    x = channel_attention(x)

    if inputs.shape[-1] == filters:
        x = layers.Add()([inputs, x])

    return x

# === Cosine Annealing Scheduler ===
def cosine_annealing(epoch, initial_lr=0.001, epochs_per_cycle=50):
    return initial_lr / 2 * (np.cos(np.pi * (epoch % epochs_per_cycle) / epochs_per_cycle) + 1)

# === Model Builder ===
def build_mnasnet_model(input_shape=(19, 19, 31), filters=33, blocks=7, expansion_factor=4, reg_strength=1e-4, dropout_rate=0.3):
    inputs = Input(shape=input_shape, name='board')

    x = layers.Conv2D(filters, (3, 3), padding='same', use_bias=False, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    for _ in range(blocks):
        x = mnasnet_block_with_attention(x, filters, (3, 3), expansion_factor)

    # Policy head
    policy = layers.Conv2D(128, (1, 1), use_bias=False, activation='relu',
                           kernel_regularizer=regularizers.l2(reg_strength))(x)
    policy = layers.BatchNormalization()(policy)
    policy = layers.Conv2D(1, (1, 1))(policy)
    policy = layers.Flatten()(policy)
    policy = layers.Activation('softmax', name='policy')(policy)

    # Value head
    value = layers.GlobalAveragePooling2D()(x)
    value = layers.Dense(80, activation='relu')(value)
    value = layers.Dropout(dropout_rate)(value)
    value = layers.Dense(1, activation='sigmoid', name='value')(value)

    model = Model(inputs=inputs, outputs=[policy, value])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
        metrics={'policy': 'categorical_accuracy', 'value': ['mse']}
    )

    return model
