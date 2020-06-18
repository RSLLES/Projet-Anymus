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
BATCH_SIZE = 6
EPOCHS = 200
SAMPLE_INTERVAL = 400

# Input shape
IMG_ROWS = 128
IMG_COLS = 128
CHANNELS = 3
IMG_SHAPE = (IMG_ROWS, IMG_COLS, CHANNELS)

# Number of filters in the first layer of G and D
GF, DF = 16, 16

# Loss weights
LAMBDA_CYCLE = 8               # Cycle-consistency loss
LAMBDA_ID = 0.1 * LAMBDA_CYCLE    # Identity loss

#Optimize
learning_rate = 0.0002
discr_factor = 0.3
OPTIMIZER = Adam(learning_rate, 0.5)
OPTIMIZER_D = Adam(learning_rate*discr_factor, 0.5)

START_EPO = 21



########################
##### Programme ########
########################

# Configure data loader
dataset_name = 'ugatit'
data_loader = DataLoader(dataset_name=dataset_name, img_res=(IMG_ROWS, IMG_COLS))



#
# Construction des réseaux
#

#
# Briques de base
#

def conv2d(layer_input, filters, f_size=4, strides=2, normalization=True):
    """Layer de base pour downsamplé"""
    l = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
    if normalization:
        l = InstanceNormalization()(l)
    l = LeakyReLU(alpha=0.2)(l)
    return l

def dilated_conv2d(layer_input, filters, f_size = 3, d = 2):
    l = Conv2D(filters, kernel_size=f_size, dilation_rate=d, padding='same')(layer_input)
    l = InstanceNormalization()(l)
    l = LeakyReLU(alpha=0.2)(l)
    return l


def resnet(layer_input, filters, f_size=3):
    l = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(layer_input)
    l = LeakyReLU(alpha=0.2)(l)
    l = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(l)

    l = Concatenate()([l, layer_input])

    l = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')(l)
    l = LeakyReLU(alpha=0.2)(l)
    l = InstanceNormalization()(l)
    return l

def deconv2d(layer_input, skip_input, filters, f_size=4):
    """Layers used during upsampling"""
    u = UpSampling2D(size=2)(layer_input)
    u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
    u = InstanceNormalization()(u)
    for _ in range(3):
        u = resnet(u, filters)
    u = Concatenate()([u, skip_input])
    return u

# 
# Réseaux 
# 

def build_discriminator(name=""):
    # Entrée
    img = Input(shape=IMG_SHAPE)

    # Diminution
    d1 = conv2d(img, DF,f_size=7 ,normalization=False)
    d2 = conv2d(d1, 2*DF)
    d2 = conv2d(d1, 4*DF, strides=1)

    # Etude avec de la convolution dilated
    d3 = dilated_conv2d(d2, 8*DF, d=2)
    d4 = dilated_conv2d(d3, 8*DF, d=4)
    d5 = dilated_conv2d(d4, 8*DF, d=8)

    # Skip et fin du réseau
    d6 = Concatenate()([d2,d5])
    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d6)

    # Compilation
    model = Model(img, validity, name="discr_{}".format(name))
    model.compile(loss='mse', optimizer=OPTIMIZER_D, metrics=['accuracy'])

    return model

def build_generator(name=""):
    """U-Net Generator"""

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

    return Model(d0, output_img, name="gen_{}".format(name))


# Build and compile the discriminators
d_A = build_discriminator("A")
d_B = build_discriminator("B")

# Build the generators
g_AB = build_generator("AB")
g_BA = build_generator("BA")


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

def sample_images(epoch, batch_i, gif=False):
    if gif:
        os.makedirs('images/%s/gif' % dataset_name, exist_ok=True)
    else:
        os.makedirs('images/%s' % dataset_name, exist_ok=True)

    r, c = 2, 3

    if gif:
        imgs_A = data_loader.load_img('datasets/{}/testA/gif.jpg'.format(dataset_name))
        imgs_B = data_loader.load_img('datasets/{}/testB/gif.jpg'.format(dataset_name))
    else:
        imgs_A = data_loader.load_data(domain="A", batch_size=1, is_testing=True)
        imgs_B = data_loader.load_data(domain="B", batch_size=1, is_testing=True)

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
    if gif:
        fig.savefig("images/%s/gif/%d_%d.png" % (dataset_name, epoch, batch_i))
    else:
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

for epoch in range(START_EPO,EPOCHS):
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
            sample_images(epoch, batch_i, gif=True)
            save()
