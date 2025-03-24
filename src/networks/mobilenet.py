# src/architectures/mobilenet_like.py

import tensorflow as tf
from tensorflow.keras import layers, Model, Input

def SE_Block(inputs, ratio=16):
    """Squeeze-and-Excitation block."""
    filters = inputs.shape[-1]
    se = layers.GlobalAveragePooling2D()(inputs)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    x = layers.Multiply()([inputs, se])
    return x

def dual_path_block(inputs, filters):
    """Dual depthwise convolution block with SE and residual."""
    x = layers.Conv2D(128, (1, 1), use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    path1 = layers.DepthwiseConv2D((3, 3), padding='same', use_bias=False)(x)
    path2 = layers.DepthwiseConv2D((5, 5), padding='same', use_bias=False)(x)

    x = layers.Concatenate()([path1, path2])
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, (1, 1), use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    x = SE_Block(x)

    return layers.Add()([inputs, x])

def build_mobilenet_like_model(input_shape=(19, 19, 31), filters=32, blocks=5):
    inputs = Input(shape=input_shape, name='board')

    x = layers.Conv2D(filters, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    for _ in range(blocks):
        x = dual_path_block(x, filters)

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
        optimizer=tf.keras.optimizers.Adam(),
        loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
        loss_weights={'policy': 1.0, 'value': 1.0},
        metrics={'policy': 'categorical_accuracy', 'value': 'mse'}
    )

    return model
