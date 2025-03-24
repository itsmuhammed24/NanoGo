# src/architectures/transformer_dyt.py

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model, Input

# === Dynamic Tanh Activation Layer ===
class DyT(layers.Layer):
    def __init__(self, channels, **kwargs):
        super(DyT, self).__init__(**kwargs)
        self.alpha = self.add_weight(shape=(1,), initializer='ones', trainable=True)
        self.gamma = self.add_weight(shape=(channels,), initializer='ones', trainable=True)
        self.beta = self.add_weight(shape=(channels,), initializer='zeros', trainable=True)

    def call(self, inputs):
        return self.gamma * tf.math.tanh(self.alpha * inputs) + self.beta

# === Transformer Encoder Block with DyT ===
def transformer_block(inputs, embed_dim, num_heads, ff_dim):
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attn_output = DyT(embed_dim)(attn_output)
    x = layers.Add()([inputs, attn_output])

    ffn_output = layers.Dense(ff_dim, activation='relu')(x)
    ffn_output = layers.Dense(embed_dim)(ffn_output)
    ffn_output = DyT(embed_dim)(ffn_output)
    x = layers.Add()([x, ffn_output])
    return x

# === Model Builder ===
def build_transformer_dyt_model(
    input_shape=(19, 19, 31),
    embed_dim=64,
    num_heads=4,
    ff_dim=128,
    num_layers=4,
    reg_strength=1e-5
):
    inputs = Input(shape=input_shape, name='board')

    # CNN to embedding
    x = layers.Conv2D(embed_dim, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(reg_strength))(inputs)
    x = layers.Reshape((19 * 19, embed_dim))(x)  # Shape: (batch, 361, embed_dim)

    # Transformer layers
    for _ in range(num_layers):
        x = transformer_block(x, embed_dim, num_heads, ff_dim)

    # Policy head (on the CLS-like token)
    policy = layers.Dense(361, activation='softmax', name='policy')(x[:, 0, :])  # assume first token like CLS

    # Value head (global pooling)
    value = layers.GlobalAveragePooling1D()(x)
    value = layers.Dense(50, activation='relu')(value)
    value = layers.Dense(1, activation='sigmoid', name='value')(value)

    model = Model(inputs=inputs, outputs=[policy, value])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
        loss_weights={'policy': 1.0, 'value': 1.0},
        metrics={'policy': 'categorical_accuracy', 'value': 'mse'}
    )

    return model
