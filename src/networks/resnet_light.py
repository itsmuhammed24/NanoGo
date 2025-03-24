# src/architectures/resnet_light.py

import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model, Input

def resnet_block(inputs, filters):
    shortcut = inputs
    x = layers.Conv2D(filters, (3, 3), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    return x

def input_residual_layer(inputs, filters):
    conv_5x5 = layers.Conv2D(filters, (5, 5), padding='same', use_bias=False)(inputs)
    conv_1x1 = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)(inputs)
    x = layers.Add()([conv_5x5, conv_1x1])
    x = layers.ReLU()(x)
    return x

def build_resnet_model(input_shape=(19, 19, 31), filters=25, blocks=6, reg_strength=1e-5):
    inputs = Input(shape=input_shape, name='board')
    x = input_residual_layer(inputs, filters)

    for _ in range(blocks):
        x = resnet_block(x, filters)

    # Policy head
    policy = layers.Conv2D(128, (1, 1), use_bias=False)(x)
    policy = layers.BatchNormalization()(policy)
    policy = layers.ReLU()(policy)
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
        metrics={'policy': 'categorical_accuracy', 'value': ['mse', 'mae']}
    )
    
    return model
