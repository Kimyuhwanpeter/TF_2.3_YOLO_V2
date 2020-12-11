# -*- coding:utf-8 -*-
import tensorflow as tf

Conv2D = tf.keras.layers.Conv2D
MaxPool2D = tf.keras.layers.MaxPool2D
LeakReLU = tf.keras.layers.LeakyReLU
BatchNorm = tf.keras.layers.BatchNormalization

class conv(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(conv, self).__init__()
        self.conv = Conv2D(**kwargs)
        self.leak_relu = LeakReLU(0.1)
        self.bn = BatchNorm()
    def call(self, inputs):
        return self.leak_relu(self.bn(self.conv(inputs)))

def Yolo_V2(input_shape=(416, 416, 3)):
    # https://89douner.tistory.com/93
    h = inputs = tf.keras.Input(input_shape)

    h = conv(filters=32,
             kernel_size=3,
             padding="same",
             use_bias=False)(h)
    h = MaxPool2D(pool_size=(2,2),
                  strides=(2,2),
                  padding="same")(h)    # [N x 208 x 208 x 32]
    
    h = conv(filters=64,
             kernel_size=3,
             padding="same",
             use_bias=False)(h)
    h = MaxPool2D(pool_size=(2,2),
                  strides=(2,2),
                  padding="same")(h)    # [N x 104 x 104 x 64]

    h = conv(filters=128,
             kernel_size=3,
             padding="same",
             use_bias=False)(h)
    h = conv(filters=64,
             kernel_size=1,
             padding="same",
             use_bias=False)(h)
    h = conv(filters=128,
             kernel_size=3,
             padding="same",
             use_bias=False)(h)
    h = MaxPool2D(pool_size=(2,2),
                  strides=(2,2),
                  padding="same")(h)    # [N x 52 x 52 x 128]

    h = conv(filters=256,
             kernel_size=3,
             padding="same",
             use_bias=False)(h)
    h = conv(filters=128,
             kernel_size=1,
             padding="same",
             use_bias=False)(h)
    h = conv(filters=256,
             kernel_size=3,
             padding="same",
             use_bias=False)(h)
    h = MaxPool2D(pool_size=(2,2),
                  strides=(2,2),
                  padding="same")(h)    # [N x 26 x 26 x 256]

    h = conv(filters=512,
             kernel_size=3,
             padding="same",
             use_bias=False)(h)
    h = conv(filters=256,
             kernel_size=1,
             padding="same",
             use_bias=False)(h)
    h = conv(filters=512,
             kernel_size=3,
             padding="same",
             use_bias=False)(h)
    h = conv(filters=256,
             kernel_size=1,
             padding="same",
             use_bias=False)(h)
    skip_connection1 = conv(filters=512,
                             kernel_size=3,
                             padding="same",
                             use_bias=False)(h)
    h = MaxPool2D(pool_size=(2,2),
                  strides=(2,2),
                  padding="same")(skip_connection1)    # [N x 13 x 13 x 512]

    h = conv(filters=1024,
             kernel_size=3,
             padding="same",
             use_bias=False)(h)
    h = conv(filters=512,
             kernel_size=1,
             padding="same",
             use_bias=False)(h)
    h = conv(filters=1024,
             kernel_size=3,
             padding="same",
             use_bias=False)(h)
    h = conv(filters=512,
             kernel_size=1,
             padding="same",
             use_bias=False)(h)
    h = conv(filters=1024,
             kernel_size=3,
             padding="same",
             use_bias=False)(h) # [N x 13 x 13 x 1024]

    h = conv(filters=1024,
             kernel_size=3,
             padding="same",
             use_bias=False)(h)
    skip_connection2 = conv(filters=512,
                           kernel_size=3,
                           padding="same",
                           use_bias=False)(h) # [N x 13 x 13 x 1024]

    h = conv(filters=64,
             kernel_size=1,
             padding="same",
             use_bias=False)(skip_connection1)
    h = reorg(h)    # [N x 13 x 13 x 256]

    h = tf.concat([h, skip_connection2], 3) # [N x 13 x 13 x 1280]

    h = conv(filters=1024,
             kernel_size=3,
             padding="same",
             use_bias=False)(h) # [N x 13 x 13 x 1024]
    h = conv(filters=5 * (20 + 5),
             kernel_size=1,
             padding="same",
             use_bias=False)(h) # [N x 13 x 13 x 125]
    
    return tf.keras.Model(inputs=inputs, outputs=h)

def reorg(inputs):
    outputs_1 = inputs[:, ::2, ::2, :]
    outputs_2 = inputs[:, ::2, 1::2, :]
    outputs_3 = inputs[:, 1::2, ::2, :]
    outputs_4 = inputs[:, 1::2, 1::2, :]

    output = tf.concat([outputs_1, outputs_2, outputs_3, outputs_4], 3)
    return output