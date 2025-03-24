# src/architectures/inception_cbam.py

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model, Input

# === CBAM Attention ===
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

# === Inception-style Block with CBAM ===
def inception_block(inputs, filters, reg_strength=1e-4):
    conv1x1 = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False, activation='swish')(inputs)
    depthwise3x3 = layers.SeparableConv2D(filters, (3, 3), padding='same', use_bias=False, activation='swish')(inputs)
    depthwise5x5 = layers.SeparableConv2D(filters, (5, 5), padding='same', use_bias=False, activation='swish')(inputs)

    x = layers.Concatenate()([conv1x1, depthwise3x3, depthwise5x5])
    x = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False, activation='swish')(x)
    x = CBAM(filters)(x)

    # Skip connection (project if needed)
    if inputs.shape[-1] != filters:
        inputs = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)(inputs)

    return layers.Add()([x, inputs])

# === Model Builder ===
def build_inception_cbam_model(input_shape=(19, 19, 31), filters=41, blocks=6, reg_strength=1e-4, dropout_rate=0.3):
    inputs = Input(shape=input_shape, name='board')

    # Initial conv layer
    x = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False, activation='swish')(inputs)
    x = inception_block(x, filters, reg_strength=reg_strength)

    for _ in range(blocks):
        x = inception_block(x, filters, reg_strength=reg_strength)

    # Policy head
    policy = layers.Conv2D(128, (1, 1), use_bias=False, activation='swish',
                           kernel_regularizer=regularizers.l2(reg_strength))(x)
    policy = layers.BatchNormalization()(policy)
    policy = layers.Conv2D(1, (1, 1))(policy)
    policy = layers.Flatten()(policy)
    policy = layers.Activation('softmax', name='policy')(policy)

    # Value head
    value = layers.GlobalAveragePooling2D()(x)
    value = layers.Dense(80, activation='swish')(value)
    value = layers.Dropout(dropout_rate)(value)
    value = layers.Dense(1, activation='sigmoid', name='value')(value)

    model = Model(inputs=inputs, outputs=[policy, value])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
        metrics={'policy': 'categorical_accuracy', 'value': ['mse', 'mae']}
    )

    return model
