from __future__ import print_function, division
import scipy

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Add, Layer, Lambda
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers.convolutional import UpSampling2D
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import tensorflow as tf
import numpy as np
import os
import keras.backend as K
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

### Constantes du programmes ###

# Sur l'entrainement
START_EPO = 0
if len(sys.argv) > 1:
    START_EPO = int(sys.argv[1])
BATCH_SIZE = 1 #FIXE ICI, SINON ERREUR DE CALCULS AVEC LE MULTIPLY
EPOCHS = 200
SAMPLE_INTERVAL = 1000

# Input shape
IMG_ROWS = 128
IMG_COLS = 128
CHANNELS = 3
IMG_SHAPE = (IMG_ROWS, IMG_COLS, CHANNELS)

# Number of filters in the first layer of G and D
GF, DF = 64, 64
N_RESNET = 4

# Loss weights
LAMBDA_AUX = 1000
LAMBDA_CYCLE = 10
LAMBDA_ID = 10    

#Optimize
learning_rate = 0.0002
OPTIMIZER = Adam(learning_rate, 0.5)



########################
##### Programme ########
########################

# Configure data loader
dataset_name = 'ugatit'
wf = "Weights/{}/".format(dataset_name)
data_loader = DataLoader(dataset_name=dataset_name, img_res=(IMG_ROWS, IMG_COLS))



#
# Construction des réseaux
#

#
# Layer d'instance utilisé qui est custom
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
# Briques
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

# 
# Réseaux 
# 

def build_generator(name=""):
    # Image input
    entree = Input(shape=IMG_SHAPE)

    # Downsampling
    g = conv2d(entree, GF, f_size=7, strides=1)
    g = conv2d(g, GF*2, f_size=3, strides=2)
    g = conv2d(g, GF*4, f_size=3, strides=2)

    # Resnet d'entrée
    for _ in range(N_RESNET):
        g = resnet(g)

    # Création de la CLASSE Activation Map (CAM) en Max et en Average
    g_mp, g_ap = GlobalMaxPooling2D()(g), GlobalAveragePooling2D()(g)

    cam_m, g_m = AuxClass()([g_mp, g])
    cam_a, g_a = AuxClass()([g_ap, g])

    cam = Concatenate()([cam_m, cam_a])
    g = Concatenate()([g_m, g_a])

    g = conv2d(g, GF*4, f_size=1, strides=1)
    heatmap = Lambda(lambda x: K.sum(x, axis=-1))(g)

    # Création des constantes pour AdaLIN plus tard
    def adalin_param(x, filters):
        l = Flatten()(x)
        for _ in range(2):
            l = Dense(filters)(l)
            l = LeakyReLU(alpha=0.2)(l)
        return Dense(filters)(l), Dense(filters)(l)
    gamma, beta = adalin_param(g, GF*4)

    # Resnet de Sorties
    for _ in range(N_RESNET):
        g = resnet_adalin(g, gamma, beta)

    # Upscaling
    g = deconv2d_adalin(g, GF*2, f_size=4)
    g = deconv2d_adalin(g, GF, f_size=4)

    # Fin du réseau
    g = Conv2D(CHANNELS, kernel_size=4, strides=1, padding='same', activation='tanh')(g)

    g_model = Model(entree, g, name="gen_{}".format(name))
    aux_model = Model(entree, cam, name="aux_gen_{}".format(name))
    heatmap_model = Model(entree, heatmap, name="heatmap_{}".format(name))

    return g_model, aux_model, heatmap_model

def build_discriminator(name=""):
    # Image input
    entree = Input(shape=IMG_SHAPE)

    # Downsampling
    d = conv2d(entree, DF, f_size=7, strides=1)
    d = conv2d(d, DF*2, f_size=3, strides=2)
    d = conv2d(d, DF*4, f_size=3, strides=2)

    # Création de la CLASSE Activation Map (CAM) en Max et en Average
    d_mp, d_ap = GlobalMaxPooling2D()(d), GlobalAveragePooling2D()(d)

    A1, A2 = AuxClass(), AuxClass()
    cam_m, d_m = A1([d_mp, d])
    cam_a, d_a = A2([d_ap, d])

    Conca1, Conca2 = Concatenate(), Concatenate()
    cam = Conca1([cam_m, cam_a])
    d = Conca2([d_m, d_a])

    d = conv2d(d, DF*4, f_size=1, strides=1)
    heatmap = Lambda(lambda x: K.sum(x, axis=-1))(d)

    # Final
    d = Conv2D(1, kernel_size=4, strides=1, padding='same')(d)

    d_model = Model(entree, d, name="disc_{}".format(name))
    d_model.compile(loss='mse', optimizer=OPTIMIZER, metrics=['accuracy'])

    aux_model = Model(entree, cam, name="aux_d_{}".format(name))
    aux_model.compile(loss='mse', optimizer=OPTIMIZER, metrics=['accuracy'])

    heatmap_model = Model(entree, heatmap, name="heatmap_{}".format(name))

    return d_model, aux_model, heatmap_model

# Build and compile the discriminators
d_A, aux_d_A, heatmap_d_A = build_discriminator("A")
d_B, aux_d_B, heatmap_d_B = build_discriminator("B")

# Build the generators
g_AB, aux_g_AB, heatmap_g_AB = build_generator("AB")
g_BA, aux_g_BA, heatmap_g_BA = build_generator("BA")


#Load
def load():
    """Sauvegarde les poids deja calculés, pour pouvoir reprendre les calculs plus tard si jamais"""
    os.makedirs(wf, exist_ok=True)
    if (os.path.isfile(wf + "d_A.h5") and os.path.isfile(wf + "g_AB.h5") 
    and os.path.isfile(wf + "aux_d_A.h5") and os.path.isfile(wf + "aux_g_AB.h5") 
    and os.path.isfile(wf + "d_B.h5") and os.path.isfile(wf + "g_BA.h5")
    and os.path.isfile(wf + "aux_d_B.h5") and os.path.isfile(wf + "aux_g_BA.h5")):
        d_A.load_weights(wf + "d_A.h5")
        d_B.load_weights(wf + "d_B.h5")
        g_AB.load_weights(wf + "g_AB.h5")
        g_BA.load_weights(wf + "g_BA.h5")
        aux_d_A.load_weights(wf + "aux_d_A.h5")
        aux_d_B.load_weights(wf + "aux_d_B.h5")
        aux_g_AB.load_weights(wf + "aux_g_AB.h5")
        aux_g_BA.load_weights(wf + "aux_g_BA.h5")
        print("Weights loaded")
    else:
        print("Missing weights files detected. Starting from scratch")
load()

def build_combined():
    # Input images from both domains
    img_A = Input(shape=IMG_SHAPE)
    img_B = Input(shape=IMG_SHAPE)

    # Translate images to the other domain
    fake_B = g_AB(img_A)
    fake_A = g_BA(img_B)

    # Sortie Auxilliary classifier des générateurs
    aux_A_dans_AB = aux_g_AB(img_A)
    aux_B_dans_AB = aux_g_AB(img_B)
    aux_A_dans_BA = aux_g_BA(img_A)
    aux_B_dans_BA = aux_g_BA(img_B)

    # Translate images back to original domain
    reconstr_A = g_BA(fake_B)
    reconstr_B = g_AB(fake_A)

    # Identity mapping of images
    img_A_id = g_BA(img_A)
    img_B_id = g_AB(img_B)

    # For the combined model we will only train the generators
    d_A.trainable = False
    d_B.trainable = False
    aux_d_A.trainable = False
    aux_d_B.trainable = False

    # Discriminators determines validity of translated images
    valid_A = d_A(fake_A)
    valid_B = d_B(fake_B)

    # Combined model trains generators to fool discriminators
    model = Model(inputs=[img_A, img_B], outputs=[  valid_A, valid_B, 
                                                    reconstr_A, reconstr_B, 
                                                    img_A_id, img_B_id,
                                                    aux_A_dans_AB, aux_B_dans_BA,
                                                    aux_A_dans_BA, aux_B_dans_AB])
    model.compile(  loss=[  'mse', 'mse', 
                            'mae', 'mae',
                            'mae', 'mae',
                            'mse', 'mse',
                            'mse', 'mse'],
                    loss_weights=[  1, 1,
                                    LAMBDA_CYCLE, LAMBDA_CYCLE, 
                                    LAMBDA_ID, LAMBDA_ID,
                                    LAMBDA_AUX, LAMBDA_AUX,
                                    LAMBDA_AUX, LAMBDA_AUX ],
        optimizer=OPTIMIZER)

    return model

# Build the combined model to train generators


combined = build_combined()


#
# Lancement de lentrainement
# 

def sample_images(epoch, batch_i, gif=False):
    if gif:
        os.makedirs('images/%s/gif' % dataset_name, exist_ok=True)
    else:
        os.makedirs('images/%s' % dataset_name, exist_ok=True)

    r, c = 2, 5

    imgs_A = data_loader.load_data(domain="A", batch_size=1, is_testing=True)
    imgs_B = data_loader.load_data(domain="B", batch_size=1, is_testing=True)

    # Rescale btw 0 - 1 img initially btw -1 and 1
    def rescale(img):
        return 0.5*img + 0.5

    # Translate images to the other domain
    fake_B = rescale(g_AB.predict(imgs_A))
    fake_A = rescale(g_BA.predict(imgs_B))
    # Translate back to original domain
    reconstr_A = rescale(g_BA.predict(fake_B))
    reconstr_B = rescale(g_AB.predict(fake_A))
    # Calculate heatmap
    hm_g_A = heatmap_g_AB.predict(imgs_A)
    hm_g_B = heatmap_g_BA.predict(imgs_B)
    hm_d_A = heatmap_d_A.predict(fake_A)
    hm_d_B = heatmap_d_B.predict(fake_B)

    titles = ['Original', 'Gen', 'Translated', 'Discr', 'Reconstructed']
    fig, axs = plt.subplots(r, c, dpi=200)

    def show_row(r_i, img, hm_g, fake, hm_d, reconstr):
        def show(i,j, im, is_heatmap=False):
            if is_heatmap:
                axs[i,j].imshow(im[0,...], vmin=-1., vmax=1., cmap='RdBu_r')
            else:
                axs[i,j].imshow(im[0,...])
            axs[i,j].set_title(titles[j])
            axs[i,j].axis('off')
        # Image normale
        show(r_i, 0, img)
        # Heatmap gen
        show(r_i, 1, hm_g, is_heatmap=True)
        # Fake
        show(r_i, 2, fake)
        # Heatmap disc
        show(r_i, 3, hm_d, is_heatmap=True)
        # Reconstructed
        show(r_i, 4, reconstr)

    show_row(0, imgs_A, hm_g_A, fake_B, hm_d_B, reconstr_A)
    show_row(1, imgs_B, hm_g_B, fake_A, hm_d_A, reconstr_B)

    fig.savefig("images/%s/%d_%d.png" % (dataset_name, epoch, batch_i), dpi=200)
    plt.close()

def save():
    """Sauvegarde les poids deja calculés, pour pouvoir reprendre les calculs plus tard si jamais"""
    os.makedirs(wf, exist_ok=True)
    d_A.save_weights(wf + "d_A.h5")
    d_B.save_weights(wf + "d_B.h5")
    g_AB.save_weights(wf + "g_AB.h5")
    g_BA.save_weights(wf + "g_BA.h5")
    aux_d_A.save_weights(wf + "aux_d_A.h5")
    aux_d_B.save_weights(wf + "aux_d_B.h5")
    aux_g_AB.save_weights(wf + "aux_g_AB.h5")
    aux_g_BA.save_weights(wf + "aux_g_BA.h5")

start_time = datetime.datetime.now()

# Adversarial loss ground truths
valid = np.ones((BATCH_SIZE,) + d_A.output_shape[1:])
fake = np.zeros((BATCH_SIZE,) + d_A.output_shape[1:])

aux_valid = np.ones((BATCH_SIZE,) + aux_d_A.output_shape[1:])
aux_fake = np.zeros((BATCH_SIZE,) + aux_d_A.output_shape[1:])

for epoch in range(START_EPO,EPOCHS):
    for batch_i, (imgs_A, imgs_B) in enumerate(data_loader.load_batch(BATCH_SIZE)):

        # ----------------------
        #  Train Discriminators
        # ----------------------

        # Translate images to opposite domain
        fake_B = g_AB.predict(imgs_A)
        fake_A = g_BA.predict(imgs_B)

        # Train the discriminators auxiliary classifier
        aux_dA_loss_real = aux_d_A.train_on_batch(imgs_A, aux_valid)
        aux_dA_loss_fake = aux_d_A.train_on_batch(fake_A, aux_fake)
        aux_dA_loss = 0.5 * np.add(aux_dA_loss_real, aux_dA_loss_fake)

        aux_dB_loss_real = aux_d_B.train_on_batch(imgs_B, aux_valid)
        aux_dB_loss_fake = aux_d_B.train_on_batch(fake_B, aux_fake)
        aux_dB_loss = 0.5 * np.add(aux_dB_loss_real, aux_dB_loss_fake)

        # Total auxiliary classifier loss
        aux_d_loss = 0.5 * np.add(aux_dA_loss, aux_dB_loss)


        # Train the discriminators (original images = real / translated = Fake)
        dA_loss_real = d_A.train_on_batch(imgs_A, valid)
        dA_loss_fake = d_A.train_on_batch(fake_A, fake)
        dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

        dB_loss_real = d_B.train_on_batch(imgs_B, valid)
        dB_loss_fake = d_B.train_on_batch(fake_B, fake)
        dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

        # Total disciminator loss
        d_loss = 0.5 * np.add(dA_loss, dB_loss)


        # ------------------
        #  Train Generators
        # ------------------

        # Train the generators
        g_loss = combined.train_on_batch([imgs_A, imgs_B],[ valid, valid,
                                                            imgs_A, imgs_B,
                                                            imgs_A, imgs_B,
                                                            aux_valid, aux_valid,
                                                            aux_fake, aux_fake])

        elapsed_time = datetime.datetime.now() - start_time

        # Plot the progress
        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [AuxD loss : %f, acc : %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                                                                % ( epoch, EPOCHS,
                                                                    batch_i, data_loader.n_batches,
                                                                    d_loss[0], 100*d_loss[1],
                                                                    aux_d_loss[0], 100*aux_d_loss[1],
                                                                    g_loss[0],
                                                                    np.mean(g_loss[1:3]),
                                                                    np.mean(g_loss[3:5]),
                                                                    np.mean(g_loss[5:6]),
                                                                    elapsed_time))

        # If at save interval => save generated image samples
        if batch_i % SAMPLE_INTERVAL == 0:
            sample_images(epoch, batch_i)
            #sample_images(epoch, batch_i, gif=True)
            save()