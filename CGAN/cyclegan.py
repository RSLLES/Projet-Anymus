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
BATCH_SIZE = 3
EPOCHS = 200
SAMPLE_INTERVAL = 200

# Input shape
IMG_ROWS = 256
IMG_COLS = 256
CHANNELS = 3
IMG_SHAPE = (IMG_ROWS, IMG_COLS, CHANNELS)

# Number of filters in the first layer of G and D
GF, DF = 32, 32

# Loss weights
LAMBDA_CYCLE = 10               # Cycle-consistency loss
LAMBDA_ID = 0.1 * LAMBDA_CYCLE    # Identity loss

#Optimize
learning_rate = 0.0002
discr_factor = 0.3
OPTIMIZER = Adam(learning_rate, 0.5)
OPTIMIZER_D = Adam(learning_rate*discr_factor, 0.5)

#Folders
wf = "Weights/{}/".format(dataset_name)



########################
##### Programme ########
########################

# Configure data loader
dataset_name = 'landscape2myazaki'
data_loader = DataLoader(dataset_name=dataset_name, img_res=(IMG_ROWS, IMG_COLS))

#
# Construction des réseaux
#

def build_discriminator(name=""):
    """ Format classique d'un discriminateur"""

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

def build_generator():
    """U-Net Generator"""

    def conv2d(layer_input, filters, f_size=4):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
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
d_A = build_discriminator()
d_B = build_discriminator()

# Build the generators
g_AB = build_generator()
g_BA = build_generator()


#Load
def load():
    """Sauvegarde les poids deja calculés, pour pouvoir reprendre les calculs plus tard si jamais"""
    os.makedirs(wf, exist_ok=True)
    if (os.path.isfile(wf + "d_A.h5") and os.path.isfile(wf + "g_AB.h5") 
    and os.path.isfile(wf + "d_B.h5") and os.path.isfile(wf + "g_BA.h5")):
        d_A.load_weights(wf + "d_A.h5")
        d_B.load_weights(wf + "d_B.h5")
        g_AB.load_weights(wf + "g_AB.h5")
        g_BA.load_weights(wf + "g_BA.h5")
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
        imgs_A = data_loader.load_img('datasets/face2manga/testA/gif.jpg')
        imgs_B = data_loader.load_img('datasets/face2manga/testB/gif.jpg')
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
    os.makedirs(wf, exist_ok=True)
    d_A.save_weights(wf + "d_A.h5")
    d_B.save_weights(wf + "d_B.h5")
    g_AB.save_weights(wf + "g_AB.h5")
    g_BA.save_weights(wf + "g_BA.h5")

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
            sample_images(epoch, batch_i, gif=True)
            save()
