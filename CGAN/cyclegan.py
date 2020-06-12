from __future__ import print_function, division
import scipy

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Add
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers.convolutional import UpSampling2D
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

### Constantes du programmes ###

# Sur l'entrainement
BATCH_SIZE = 15
EPOCHS = 200
SAMPLE_INTERVAL = 100

# Input shape
IMG_ROWS = 128
IMG_COLS = 128
CHANNELS = 3
IMG_SHAPE = (IMG_ROWS, IMG_COLS, CHANNELS)

# Number of filters in the first layer of G and D
GF, DF = 64, 32

# Loss weights
LAMBDA_CYCLE = 10               # Cycle-consistency loss
LAMBDA_ID = 0.1 * LAMBDA_CYCLE    # Identity loss

#Optimize
learning_rate = 0.0002
discr_factor = 0.25
OPTIMIZER = Adam(learning_rate, 0.5)
OPTIMIZER_D = Adam(learning_rate*discr_factor, 0.5)



########################
##### Programme ########
########################

# Configure data loader
dataset_name = 'face2manga'
data_loader = DataLoader(dataset_name=dataset_name, img_res=(IMG_ROWS, IMG_COLS))



#
# Construction des réseaux
#

def build_discriminator_improved(name=""):
    def d_layer(layer_input, filters, f_size=4, normalization=True, drate=1):
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        if drate > 1:
            dprime = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            dprime = Conv2D(filters, kernel_size=f_size, dilation_rate=drate, padding='same')(dprime)
            d = Concatenate()([d,dprime])
        d = LeakyReLU(alpha=0.2)(d)
        if normalization:
            d = InstanceNormalization()(d)
        return d

    img = Input(shape=IMG_SHAPE)

    d1 = d_layer(img, DF, drate=8, normalization=False)
    d2 = d_layer(d1, DF*2, drate=4)
    d3 = d_layer(d2, DF*4, drate=2)
    d4 = d_layer(d3, DF*8)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    model = Model(img, validity, name=name)
    model.compile(loss='mse', optimizer=OPTIMIZER_D, metrics=['accuracy'])

    return model

def build_discriminator(name=""):

    def d_layer(layer_input, filters, f_size=4, normalization=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if normalization:
            d = InstanceNormalization()(d)
        return d

    img = Input(shape=IMG_SHAPE)

    d1 = d_layer(img, DF, normalization=False)
    d2 = d_layer(d1, DF*2)
    d3 = d_layer(d2, DF*4)
    d4 = d_layer(d3, DF*8)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    model = Model(img, validity, name=name)
    model.compile(loss='mse', optimizer=OPTIMIZER_D, metrics=['accuracy'])

    return model

def build_generator_improved():
    def conv2d(layer_input, filters, s=2, f_size=4):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=s, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)
        return d
    def tconv2d(layer_input, filters, f_size=4, s=2):
        l = Conv2DTranspose(filters, kernel_size=f_size, strides=s, padding="same")(layer_input)
        l = InstanceNormalization(axis=-1)(l)
        l = LeakyReLU(alpha=0.2)(l)
        return l
    def resnet(layer_input, filters):
        r = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(layer_input)
        r = LeakyReLU(alpha=0.2)(r)
        r = Conv2D(layer_input.shape[-1], kernel_size=(3,3), strides=1, padding='same')(r)
        r = Add()([layer_input, r])
        r = LeakyReLU(alpha=0.2)(r)
        r = InstanceNormalization()(r)
        return r
    def resnet3(layer_input, filters):
        for _ in range(3):
            layer_input = resnet(layer_input, filters)
        return layer_input
    def downsampling(layer_input, filters, s=2):
        d = conv2d(layer_input, filters, s=s)
        d = resnet3(d, filters)
        return d
    def upsampling(layer_input, skip_layer, filters):
        u = tconv2d(layer_input, skip_layer.shape[-1])
        u = Add()([skip_layer, u])
        u = resnet3(u,filters)
        return u
    #Input
    input = Input(shape=IMG_SHAPE)
    d0 = conv2d(input, GF, s=1)
    #Downsampling
    d1 = downsampling(d0, GF*2)
    d2 = downsampling(d1, GF*4)
    d3 = downsampling(d2, GF*8)
    #Upsampling
    u1 = upsampling(d3, d2, GF*4)
    u2 = upsampling(u1, d1, GF*2)
    u3 = tconv2d(u2, GF)
    output = Conv2D(CHANNELS, kernel_size=4, strides=1, padding='same', activation='tanh')(u3)

    return Model(input, output)

def build_generator_improved_own():
    """U-Net Generator"""

    def conv2d(layer_input, filters, f_size=4, drate=1):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        if drate > 1:
            d = Conv2D(filters, kernel_size=f_size, strides=1, dilation_rate=drate, padding='same')(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, drate=1, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        if (drate > 1):
            u = Conv2D(filters, kernel_size=f_size, strides=1, dilation_rate=drate, padding='same')(u)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        else:
            u = Conv2D(filters, kernel_size=f_size, strides=1, activation='relu', padding='same')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = InstanceNormalization()(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    d0 = Input(shape=IMG_SHAPE)

    # Downsampling
    d1 = conv2d(d0, GF)
    d2 = conv2d(d1, GF*2, drate=2)
    d3 = conv2d(d2, GF*4, drate=4)
    d4 = conv2d(d3, GF*8, drate=4)

    # Upsampling
    u1 = deconv2d(d4, d3, GF*4, drate=4)
    u2 = deconv2d(u1, d2, GF*2, drate=2)
    u3 = deconv2d(u2, d1, GF)

    u4 = UpSampling2D(size=2)(u3)
    output_img = Conv2D(CHANNELS, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

    return Model(d0, output_img)

def build_generator():
    """U-Net Generator"""

    def conv2d(layer_input, filters, f_size=4):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0.3):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = InstanceNormalization()(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    d0 = Input(shape=IMG_SHAPE)

    # Downsampling
    d1 = conv2d(d0, GF)
    d2 = conv2d(d1, GF*2)
    d3 = conv2d(d2, GF*4)
    d4 = conv2d(d3, GF*8)

    # Upsampling
    u1 = deconv2d(d4, d3, GF*4)
    u2 = deconv2d(u1, d2, GF*2)
    u3 = deconv2d(u2, d1, GF)

    u4 = UpSampling2D(size=2)(u3)
    output_img = Conv2D(CHANNELS, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

    return Model(d0, output_img)


# Build and compile the discriminators
d_A = build_discriminator_improved()
d_B = build_discriminator_improved()

# Build the generators
g_AB = build_generator()
g_BA = build_generator()


#Load
def load():
    """Sauvegarde les poids deja calculés, pour pouvoir reprendre les calculs plus tard si jamais"""
    if (os.path.isfile("Weights/d_A.h5") and os.path.isfile("Weights/g_AB.h5") 
    and os.path.isfile("Weights/d_B.h5.") and os.path.isfile("Weights/g_BA.h5")):
        d_A.load_weights("Weights/d_A.h5")
        d_B.load_weights("Weights/d_B.h5")
        g_AB.load_weights("Weights/g_AB.h5")
        g_BA.load_weights("Weights/g_BA.h5")
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
    # Translate images back to original domain
    reconstr_A = g_BA(fake_B)
    reconstr_B = g_AB(fake_A)
    # Identity mapping of images
    img_A_id = g_BA(img_A)
    img_B_id = g_AB(img_B)

    # For the combined model we will only train the generators
    d_A.trainable = False
    d_B.trainable = False

    # Discriminators determines validity of translated images
    valid_A = d_A(fake_A)
    valid_B = d_B(fake_B)

    # Combined model trains generators to fool discriminators
    model = Model(inputs=[img_A, img_B],outputs=[ valid_A, valid_B, reconstr_A, reconstr_B, img_A_id, img_B_id])
    model.compile(  loss=['mse', 'mse', 'mae', 'mae','mae', 'mae'],
                    loss_weights=[1, 1,LAMBDA_CYCLE, LAMBDA_CYCLE, LAMBDA_ID, LAMBDA_ID ],
        optimizer=OPTIMIZER)

    return model

# Build the combined model to train generators
combined = build_combined()


#
# Lancement de lentrainement
# 

def sample_images(epoch, batch_i):
    os.makedirs('images/%s' % dataset_name, exist_ok=True)
    r, c = 2, 3

    imgs_A = data_loader.load_data(domain="A", batch_size=1, is_testing=True)
    imgs_B = data_loader.load_data(domain="B", batch_size=1, is_testing=True)

    # Demo (for GIF)
    #imgs_A = self.data_loader.load_img('datasets/apple2orange/testA/n07740461_1541.jpg')
    #imgs_B = self.data_loader.load_img('datasets/apple2orange/testB/n07749192_4241.jpg')

    # Translate images to the other domain
    fake_B = g_AB.predict(imgs_A)
    fake_A = g_BA.predict(imgs_B)
    # Translate back to original domain
    reconstr_A = g_BA.predict(fake_B)
    reconstr_B = g_AB.predict(fake_A)

    gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    titles = ['Original', 'Translated', 'Reconstructed']
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i, j].set_title(titles[j])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("images/%s/%d_%d.png" % (dataset_name, epoch, batch_i))
    plt.close()
def save():
    """Sauvegarde les poids deja calculés, pour pouvoir reprendre les calculs plus tard si jamais"""
    d_A.save_weights("Weights/d_A.h5")
    d_B.save_weights("Weights/d_B.h5")
    g_AB.save_weights("Weights/g_AB.h5")
    g_BA.save_weights("Weights/g_BA.h5")

start_time = datetime.datetime.now()

# Adversarial loss ground truths
valid = np.ones((BATCH_SIZE,) + d_A.output_shape[1:])
fake = np.zeros((BATCH_SIZE,) + d_A.output_shape[1:])

for epoch in range(EPOCHS):
    for batch_i, (imgs_A, imgs_B) in enumerate(data_loader.load_batch(BATCH_SIZE)):

        # ----------------------
        #  Train Discriminators
        # ----------------------

        # Translate images to opposite domain
        fake_B = g_AB.predict(imgs_A)
        fake_A = g_BA.predict(imgs_B)

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
        g_loss = combined.train_on_batch([imgs_A, imgs_B],[valid, valid,imgs_A, imgs_B,imgs_A, imgs_B])

        elapsed_time = datetime.datetime.now() - start_time

        # Plot the progress
        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                                                                % ( epoch, EPOCHS,
                                                                    batch_i, data_loader.n_batches,
                                                                    d_loss[0], 100*d_loss[1],
                                                                    g_loss[0],
                                                                    np.mean(g_loss[1:3]),
                                                                    np.mean(g_loss[3:5]),
                                                                    np.mean(g_loss[5:6]),
                                                                    elapsed_time))

        # If at save interval => save generated image samples
        if batch_i % SAMPLE_INTERVAL == 0:
            sample_images(epoch, batch_i)
            save()
