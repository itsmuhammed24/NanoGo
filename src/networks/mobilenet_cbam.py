# src/architectures/cbam_bottleneck.py

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model, Input

# === CBAM Layer ===
class CBAM(tf.keras.layers.Layer):
    def __init__(self, filters, ratio=8):
        super(CBAM, self).__init__()
        self.filters = filters
        self.ratio = ratio
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.global_max_pool = layers.GlobalMaxPooling2D()
        self.dense1 = layers.Dense(filters // ratio, activation='relu', use_bias=False)
        self.dense2 = layers.Dense(filters, activation='sigmoid', use_bias=False)
        self.conv2d = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, x):
        avg_pool = self.dense2(self.dense1(self.global_avg_pool(x)))
        max_pool = self.dense2(self.dense1(self.global_max_pool(x)))
        channel_attention = tf.keras.activations.sigmoid(avg_pool + max_pool)
        channel_attention = tf.reshape(channel_attention, (-1, 1, 1, self.filters))
        x = x * channel_attention

        spatial_attention = self.conv2d(tf.reduce_mean(x, axis=-1, keepdims=True))
        x = x * spatial_attention
        return x

# === Bottleneck block with CBAM ===
def bottleneck_block(x, dropout_rate=0.4, reg_strength=2e-4, expand_ratio=4):
    input_channels = x.shape[-1]
    expand_channels = input_channels * expand_ratio

    m = layers.Conv2D(expand_channels, (1, 1), use_bias=False, kernel_regularizer=regularizers.l2(reg_strength))(x)
    m = layers.BatchNormalization()(m)
    m = layers.Activation('swish')(m)

    m = layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False)(m)
    m = layers.BatchNormalization()(m)
    m = layers.Activation('swish')(m)

    m = layers.Conv2D(input_channels, (1, 1), use_bias=False, kernel_regularizer=regularizers.l2(reg_strength))(m)
    m = layers.BatchNormalization()(m)

    m = CBAM(input_channels)(m)
    m = layers.Dropout(dropout_rate)(m)

    return layers.Add()([x, m])

# === Cosine Annealing Scheduler ===
def cosine_annealing(epoch, epochs, lr_min=1e-5, lr_max=5e-5):
    import numpy as np
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / epochs))

# === Model Builder ===
def build_cbam_bottleneck_model(input_shape=(19, 19, 31), filters=64, num_blocks=6, dropout_rate=0.4, reg_strength=2e-4):
    inputs = Input(shape=input_shape, name="board")

    # Initial conv
    x = layers.Conv2D(filters, (3, 3), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)

    # Stack of bottleneck blocks
    for _ in range(num_blocks):
        x = bottleneck_block(x, dropout_rate=dropout_rate, reg_strength=reg_strength)

    # Policy head
    policy = layers.Conv2D(128, (1, 1), use_bias=False)(x)
    policy = layers.BatchNormalization()(policy)
    policy = layers.Activation('relu')(policy)
    policy = layers.Conv2D(1, (1, 1))(policy)
    policy = layers.Flatten()(policy)
    policy = layers.Activation('softmax', name='policy')(policy)

    # Value head
    value = layers.GlobalAveragePooling2D()(x)
    value = layers.Dense(50, activation='relu')(value)
    value = layers.Dense(1, activation='sigmoid', name='value')(value)

    model = Model(inputs=inputs, outputs=[policy, value])
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(),
        loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
        loss_weights={'policy': 1.0, 'value': 1.0},
        metrics={'policy': 'categorical_accuracy', 'value': 'mse'}
    )
    return model
