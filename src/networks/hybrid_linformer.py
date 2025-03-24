# src/architectures/cnn_linformer_se.py

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model, Input
import numpy as np

# === Residual Block ===
def residual_block(x, filters):
    skip = x
    x = layers.Conv2D(filters, (3, 3), padding='same', use_bias=False, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same', use_bias=False, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    return layers.Add()([x, skip])

# === SE Block ===
def SE_Block(inputs, ratio=16):
    filters = inputs.shape[-1]
    se = layers.GlobalAveragePooling2D()(inputs)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    return layers.Multiply()([inputs, se])

# === Linformer-style Attention Block ===
def linformer_block(inputs, embed_dim=32, num_heads=2):
    seq_len = inputs.shape[1] * inputs.shape[2]
    x = layers.Reshape((seq_len, inputs.shape[-1]))(inputs)
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)(x, x)
    attn = layers.Add()([x, attn])
    attn = layers.LayerNormalization()(attn)
    x = layers.Dense(embed_dim, activation='relu')(attn)
    x = layers.Dense(inputs.shape[-1])(x)
    x = layers.Add()([attn, x])
    x = layers.LayerNormalization()(x)
    return layers.Reshape((19, 19, inputs.shape[-1]))(x)

# === Cosine Annealing LR Schedule ===
def cosine_annealing(epoch, initial_lr=0.001, epochs_per_cycle=50):
    return initial_lr / 2 * (np.cos(np.pi * (epoch % epochs_per_cycle) / epochs_per_cycle) + 1)

# === Model Builder ===
def build_cnn_linformer_se_model(input_shape=(19, 19, 31), filters=32, blocks=4, embed_dim=32, num_heads=2, reg_strength=1e-4, dropout_rate=0.3):
    inputs = Input(shape=input_shape, name='board')

    x = layers.Conv2D(filters, (3, 3), padding='same', use_bias=False, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    for _ in range(blocks):
        x = residual_block(x, filters)
        x = SE_Block(x)

    x = linformer_block(x, embed_dim=embed_dim, num_heads=num_heads)

    # Policy head
    policy = layers.Conv2D(32, (1, 1), use_bias=False, activation='relu',
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
