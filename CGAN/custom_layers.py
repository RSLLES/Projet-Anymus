import tensorflow as tf
from keras.layers import Conv2D, Add, Layer
from keras.layers.convolutional import UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

#
# Layer customisés pour le problème
#


class AdaLIN(Layer):
    def __init__(self, smoothing=True, eps = 1e-5):
        super(AdaLIN, self).__init__()
        self.smoothing = smoothing
        self.eps = eps

    def build(self, input_shape):
        if (len(input_shape) != 3):
            raise ValueError("Il faut donner dans l'ordre le layer, gamma et beta")
        self.ch = input_shape[0][-1]
        self.rho = self.add_weight(shape=(self.ch,), initializer="ones", trainable=True, name="rho",
                        constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))

    def call(self, inputs):
        x, gamma, beta = inputs[0], inputs[1], inputs[2]
        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + self.eps))
        ln_mean, ln_sigma = tf.nn.moments(x, axes=[1, 2, 3], keepdims=True)
        x_ln = (x - ln_mean) / (tf.sqrt(ln_sigma + self.eps))
        if self.smoothing :
            self.rho = tf.clip_by_value(self.rho - tf.constant(0.1), 0.0, 1.0)
        x_hat = self.rho * x_ins + (1 - self.rho) * x_ln
        x_hat = x_hat * gamma + beta
        return x_hat

class AdaLIN_simple(Layer):
    """Meme structure que précédemment mais en fixant gamma et beta a respectivement 1 et 0"""
    def __init__(self, smoothing=True, eps = 1e-5):
        super(AdaLIN_simple, self).__init__()
        self.smoothing = smoothing
        self.eps = eps
        
    def build(self, input_shape):
        self.ch = input_shape[-1]
        self.gamma = self.add_weight(shape=(self.ch,), initializer="ones", trainable=True, name="gamma",)
        self.beta = self.add_weight(shape=(self.ch,), initializer="zeros", trainable=True, name="beta",)
        self.rho = self.add_weight(shape=(self.ch,), initializer="zeros", trainable=True, name="rho", 
                        constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))

    def call(self, inputs):
        x = inputs
        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + self.eps))
        ln_mean, ln_sigma = tf.nn.moments(x, axes=[1, 2, 3], keepdims=True)
        x_ln = (x - ln_mean) / (tf.sqrt(ln_sigma + self.eps))
        if self.smoothing :
            self.rho = tf.clip_by_value(self.rho - tf.constant(0.1), 0.0, 1.0)
        x_hat = self.rho * x_ins + (1 - self.rho) * x_ln
        x_hat = x_hat * self.gamma + self.beta
        return x_hat

class AuxClass(Layer):
    """Implémente ma version du classifier auxiliaire"""
    def __init__(self):
        super(AuxClass, self).__init__()

    def build(self, input_shape):
        if (len(input_shape) != 2):
            raise ValueError("Input should be pooled layer then normal layer in a list")
        self.w = self.add_weight(shape=(input_shape[0][-1], 1),initializer="random_normal",trainable=True,name="w")
        self.b = self.add_weight(shape=(1,), initializer="random_normal", trainable=True, name="b")
        super(AuxClass, self).build(input_shape)

    def call(self, inputs):
        r1 = tf.matmul(inputs[0], self.w) + self.b
        w = tf.gather(tf.transpose(tf.nn.bias_add(self.w, self.b)), 0)
        r2 = tf.multiply(inputs[1],w)
        # Pour debugger
        #output_shapes = self.compute_output_shape([K.int_shape(i) for i in inputs])
        #r1.set_shape(output_shapes[0])
        #r2.set_shape(output_shapes[1])
        return [r1,r2]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], 1), input_shape[1]]


#
# Filtres de bases
#

def conv2d(layer_input, filters, f_size=4, strides=2, normalization=True):
    """Layer de base pour downsamplé"""
    l = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same')(layer_input)
    if normalization:
        l = InstanceNormalization()(l)
    l = LeakyReLU(alpha=0.2)(l)
    return l

def deconv2d_adalin(layer_input, filters, f_size=4):
    """Layers used during upsampling"""
    u = UpSampling2D(size=2)(layer_input)
    u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
    u = AdaLIN_simple()(u)
    return u

def resnet(layer_input, f_size=3):
    filters = layer_input.shape[-1]
    l = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(layer_input)
    l = InstanceNormalization()(l)
    l = LeakyReLU(alpha=0.2)(l)
    l = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(l)
    l = InstanceNormalization()(l)
    return Add()([l, layer_input])

def resnet_adalin(layer_input, gamma, beta, f_size=3):
    filters = layer_input.shape[-1]
    l = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(layer_input)
    l = AdaLIN()([l, gamma, beta])
    l = LeakyReLU(alpha=0.2)(l)
    l = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(l)
    l = AdaLIN()([l, gamma, beta])
    return Add()([l, layer_input])