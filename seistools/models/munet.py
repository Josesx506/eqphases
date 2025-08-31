import os

import keras.backend as K
import keras.saving
import tensorflow as tf
from keras import layers
from keras.layers import (Activation, BatchNormalization, Concatenate, Conv1D,
                          Conv1DTranspose, Dense, Dropout, Input, Layer,
                          MaxPooling1D, UpSampling1D)
from tensorflow.keras.models import Model

from seistools.utils import get_repo_dir

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

@keras.saving.register_keras_serializable()
class RepeatElementsLayer(Layer):
    def __init__(self, rep, axis=2, **kwargs):
        super(RepeatElementsLayer, self).__init__(**kwargs)
        self.rep = rep
        self.axis = axis
    
    def call(self, inputs):
        return tf.repeat(inputs, self.rep, axis=self.axis)
    
    def get_config(self):
        config = super(RepeatElementsLayer, self).get_config()
        config.update({"rep": self.rep, "axis": self.axis})
        return config

@keras.saving.register_keras_serializable()
class AbsLayer(layers.Layer):
    def call(self, inputs):
        return tf.abs(inputs)

# ----------------------------------- Functions for building the model -----------------------------------
def conv1D_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv1D(
        filters=n_filters,
        kernel_size=(kernel_size),
        kernel_initializer="he_normal",
        padding="same",
    )(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv1D(
        filters=n_filters,
        kernel_size=(kernel_size),
        kernel_initializer="he_normal",
        padding="same",
    )(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def lta_block(input_tensor, n_filters, batchnorm=True, lname=""):
    """
    This reduces the unstable gradients when training the event detector
    """
    # Use 100 kernels for the long-term average which is equivalent to 1s of data at 100Hz
    x = Conv1D(n_filters, 100, 1, padding="causal")(input_tensor)
    # Use Average pooling to compute the average after convolution.
    # I used 8 to match the output of the first upsampling which is 75
    x = layers.AveragePooling1D(8)(x)
    x = AbsLayer()(x)
    # x = layers.Lambda(lambda x: tf.abs(x), output_shape=x.shape[1:], arguments={"tf":tf})(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu", name=lname)(x)
    return x


def repeat_elem(tensor, rep):
    # lambda function to repeat Repeats the elements of a tensor along an axis by a factor of rep.
    # If tensor has shape (None, 256,256,3), lambda will return a tensor of shape
    # (None, 256,256,6), if specified axis=3 and rep=2.

    # return layers.Lambda(
    #     lambda x, repnum: K.repeat_elements(x, repnum, axis=2),
    #     arguments={"repnum": rep},
    #     output_shape=tensor.shape[1:]
    # )(tensor)
    # return layers.Lambda(
    #     lambda x: tf.repeat(x, rep, axis=2), arguments={"tf":tf},
    #     output_shape=tf.TensorShape([tensor.shape[1], tensor.shape[2] * rep])
    # )(tensor)

    return RepeatElementsLayer(rep, axis=2)(tensor)


def gating_signal(input, out_size, batchnorm=False):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = layers.Conv1D(out_size, 1, padding="same")(input)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def attention_block(x, gating, inter_shape, lname=""):
    """
    1D attention block modified after arXiv:1804.03999v3
    """
    shape_x = x.shape
    shape_g = gating.shape

    # Getting the x signal to the same shape as the gating signal
    theta_x = layers.Conv1D(inter_shape, 1, strides=1, padding="same")(x)  # 16
    shape_theta_x = theta_x.shape

    # Getting the gating signal to the same number of filters as the inter_shape
    phi_g = layers.Conv1D(inter_shape, 1, padding="same")(gating)
    upsample_g = layers.Conv1DTranspose(
        inter_shape, 3, strides=(shape_theta_x[1] // shape_g[1]), padding="same"
    )(
        phi_g
    )  # 16

    concat_xg = layers.Add()([upsample_g, theta_x])
    act_xg = layers.Activation("relu")(concat_xg)
    psi = layers.Conv1D(1, 1, padding="same")(act_xg)
    sigmoid_xg = layers.Activation("sigmoid")(psi)
    shape_sigmoid = sigmoid_xg.shape
    upsample_psi = layers.UpSampling1D(size=(shape_x[1] // shape_sigmoid[1]))(
        sigmoid_xg
    )  # 32

    upsample_psi = repeat_elem(upsample_psi, shape_x[2])

    y = layers.Multiply()([upsample_psi, x])

    result = layers.Conv1D(shape_x[2], 1, padding="same")(y)
    result_bn = layers.BatchNormalization(name=lname)(result)
    return result_bn


# ----------------------------------- Main model function -----------------------------------
def MUnet(input_wav, n_filters=8, dropout=0.5, batchnorm=True):
    # Define the Long-Term Average layer
    lta = lta_block(
        input_wav, n_filters=n_filters * 1, batchnorm=batchnorm, lname="LTA_block"
    )

    # contracting path
    c1 = conv1D_block(
        input_wav, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm
    )
    p1 = MaxPooling1D((2))(c1)
    p1 = Dropout(dropout * 0.5)(p1)

    c2 = conv1D_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling1D((2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv1D_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling1D((2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv1D_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling1D(pool_size=(3))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv1D_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # Bottom of the U-Net where the most important features are concatenated
    u6up = UpSampling1D(size=3)(c5)
    u6 = Concatenate()([u6up, c4])
    u6 = Dropout(dropout, name="Bottle_neck")(u6)

    # expansive path for P-phase prediction
    c6p = conv1D_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    g7p = gating_signal(c6p, n_filters * 4, batchnorm)
    at7p = attention_block(c3, g7p, n_filters * 4, lname="P0_attention")
    u7p = Conv1DTranspose(n_filters * 4, (3), strides=(2), padding="same")(c6p)
    u7p = Concatenate()([u7p, at7p])
    u7p = Dropout(dropout)(u7p)
    c7p = conv1D_block(u7p, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    g8p = gating_signal(c7p, n_filters * 2, batchnorm)
    at8p = attention_block(c2, g8p, n_filters * 2, lname="P1_attention")
    u8p = Conv1DTranspose(n_filters * 2, (3), strides=(2), padding="same")(c7p)
    u8p = Concatenate()([u8p, at8p])
    u8p = Dropout(dropout)(u8p)
    c8p = conv1D_block(u8p, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    g9p = gating_signal(c8p, n_filters * 1, batchnorm)
    at9p = attention_block(c1, g9p, n_filters * 1, lname="P_final_attention")
    u9p = Conv1DTranspose(n_filters * 1, (3), strides=(2), padding="same")(c8p)
    u9p = Concatenate()([u9p, at9p])
    u9p = Dropout(dropout)(u9p)
    c9p = conv1D_block(u9p, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    # expansive path for S-phase prediction
    c6s = conv1D_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    g7s = gating_signal(c6s, n_filters * 4, batchnorm)
    at7s = attention_block(c3, g7s, n_filters * 4, lname="S0_attention")
    u7s = Conv1DTranspose(n_filters * 4, (3), strides=(2), padding="same")(c6s)
    u7s = Concatenate()([u7s, at7s])
    u7s = Dropout(dropout)(u7s)
    c7s = conv1D_block(u7s, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    g8s = gating_signal(c7s, n_filters * 2, batchnorm)
    at8s = attention_block(c2, g8s, n_filters * 2, lname="S1_attention")
    u8s = Conv1DTranspose(n_filters * 2, (3), strides=(2), padding="same")(c7s)
    u8s = Concatenate()([u8s, at8s])
    u8s = Dropout(dropout)(u8s)
    c8s = conv1D_block(u8s, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    g9s = gating_signal(c8s, n_filters * 1, batchnorm)
    at9s = attention_block(c1, g9s, n_filters * 1, lname="S_final_attention")
    u9s = Conv1DTranspose(n_filters * 1, (3), strides=(2), padding="same")(c8s)
    u9s = Concatenate()([u9s, at9s])
    u9s = Dropout(dropout)(u9s)
    c9s = conv1D_block(u9s, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    # Concatenate the features from the first upsampling layer with the LTA layer
    e_conc = Concatenate(name="STA_LTA")([u6up, lta])
    e_drop = Dropout(dropout)(e_conc)  # Dropout some features to reduce overfitting
    e_flat = layers.Flatten()(
        e_drop
    )  # Flatten the layer for Event probaility prediction
    e_dense = Dense(64, activation="relu")(e_flat)
    e_prob = Dense(1, activation="sigmoid", name="Event")(e_dense)

    p_prob = Conv1D(1, (1), activation="sigmoid", name="P")(c9p)
    s_prob = Conv1D(1, (1), activation="sigmoid", name="S")(c9s)

    model = Model(inputs=[input_wav], outputs=[e_prob, p_prob, s_prob])
    return model


# ---------------------- Function for loading the saved model checkpoint ----------------------
munet_pth = f"{get_repo_dir()}/seistools/weights/Sorrento_FineTuneModel_best.h5"
def load_munet(path:str=munet_pth, lr:float=0.001):
    input_wav = Input((600, 3), name="waveforms")

    model = MUnet(input_wav, n_filters=8, dropout=0.05, batchnorm=True)

    # Add label smoothing to the event detection loss function to improve 
    # stability of the training process
    losses = {
        "Event": tf.keras.losses.BinaryCrossentropy(label_smoothing=0.01),
        "P": tf.keras.losses.binary_crossentropy,
        "S": tf.keras.losses.binary_crossentropy,
    }
    
    ### Load the model weights from the saved model
    model.load_weights(f"{path}")

    return model