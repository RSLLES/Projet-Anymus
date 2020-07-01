#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function, division
import scipy
import tensorflow
import tensorflow as tf
import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
import scipy

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, domain, batch_size=1, is_testing=False):
        data_type = "train%s" % domain if not is_testing else "test%s" % domain #domain prend valeur A ou B => indique classe à parcourir
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type)) #trouve le chemin fichier

        batch_images = np.random.choice(path, size=batch_size) #crée un batch aléatoire

        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            if not is_testing and np.random.random() > 0.5: #si on n'est pas sur un test set => proba 1/2 rajouter avec rotation
                img = np.fliplr(img)
            imgs.append(img)

        imgs = np.array(imgs)/127.5 - 1. #normalisation

        return imgs

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path_A = glob('./datasets/%s/%sA/*' % (self.dataset_name, data_type))
        path_B = glob('./datasets/%s/%sB/*' % (self.dataset_name, data_type))

        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        for i in range(self.n_batches-1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.imread(img_A)
                img_B = self.imread(img_B)

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B

    def load_img(self, path):
        img = self.imread(path)
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        pixels = load_img(path, target_size=self.img_res)
        return img_to_array(pixels)

#from tensorflow.keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU, ReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os

class Pix2Pix():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Configure data loader
        self.dataset_name = 'landscape2myazaki_filtered'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, n_filters, f_size=4, l_stride = 1, is_relu = True, is_norm = True, padding_type = 'same'):
            # On effectue le down sampling
            d = Conv2D(filters = n_filters, kernel_size = f_size, strides = l_stride, padding = padding_type)(layer_input)
            if is_norm:
                d = BatchNormalization(momentum=0.8)(d)
            if is_relu:
                d = ReLU()(d)
            return d

        def upconv2d(layer_input, n_filters, f_size, strides=2):
            """Layers used during upsampling"""
            u = Conv2DTranspose(n_filters, f_size, strides, padding = 'same')(layer_input)
            u = BatchNormalization(momentum=0.8)(u)
            u = ReLU()(u)
            return u

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            #u = Concatenate()([u, skip_input])
            return u

        def residual_block(y, nb_channels):
            shortcut = y

            y = Conv2D(nb_channels, kernel_size=(3, 3), strides=(1,1), padding='same')(y)
            y = BatchNormalization()(y)
            y = ReLU()(y)

            y = Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same')(y)
            y = BatchNormalization()(y)

            shortcut = Conv2D(nb_channels, kernel_size=(1, 1), strides=(1,1), padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)

            y = tensorflow.keras.layers.add([shortcut, y])
            return y

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, 7, 1)
        d2 = conv2d(d1, self.gf*2, 3, 2, False, False, 'same')
        d3 = conv2d(d2, self.gf*2, 3, 1)
        d4 = conv2d(d3, self.gf*4, 3, 2, False, False, 'same')
        d5 = conv2d(d4, self.gf*4, 3, 1)

        # Residual blocks 

        r1 = residual_block(d5, self.gf * 4)
        r2 = residual_block(r1, self.gf * 4)
        r3 = residual_block(r2, self.gf * 4)
        r4 = residual_block(r3, self.gf * 4)
        r5 = residual_block(r4, self.gf * 4)
        r6 = residual_block(r5, self.gf * 4)
        r7 = residual_block(r6, self.gf * 4)
        r8 = residual_block(r7, self.gf * 4)

        #Upsampling
        #u1 = upconv2d(r4, self.gf*2, 3)
        #u2 = upconv2d(u1, self.gf, 3)
        #d7 = conv2d(u2, 3, 7, 1)
        #output_img = tensorflow.keras.activations.tanh(d7)

        u1 = deconv2d(r8, r7, self.gf*4)
        u2 = deconv2d(u1, r7, self.gf*4)
        #u3 = deconv2d(u2, d4, self.gf*4)
        #u4 = deconv2d(u3, d3, self.gf*2)
        #u5 = deconv2d(u4, d2, self.gf*2)
        #u6 = deconv2d(u5, d1, self.gf)
        #u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u2)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, l_stride = 1, is_norm = True, is_relu = False):
            d = Conv2D(filters, kernel_size=f_size, strides=l_stride, padding='same')(layer_input)
            if is_norm:
                d = BatchNormalization(momentum=0.8)(d)
            if is_relu:
                d = LeakyReLU(alpha=0.2)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, f_size = 3, l_stride = 1, is_norm = False, is_relu = True)
        d2 = d_layer(d1, self.df*2, f_size = 3, l_stride = 2, is_norm = False, is_relu = True)
        d3 = d_layer(d2, self.df*4, f_size = 3, l_stride = 1, is_norm = True, is_relu = True)
        d4 = d_layer(d3, self.df*4, f_size = 3, l_stride = 2, is_norm = False, is_relu = True)
        d5 = d_layer(d4, self.df*8, f_size = 3, l_stride = 1, is_norm = True, is_relu = True)
        d6 = d_layer(d5, self.df*8, f_size = 3, l_stride = 1, is_norm = True, is_relu = True)
        d7 = d_layer(d6, 1, f_size = 3, l_stride = 4, is_norm = False, is_relu = False)

        validity = d7

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=50):
        start_time = datetime.datetime.now()
        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)

                # entrainer plus lentement 
            
                if batch_i % 1 == 0:
                    # Train the discriminators (original images = real / generated = Fake)
                    d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                    d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time))


                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)

    def sample_images(self, epoch, batch_i):
        os.makedirs('C:/Users/eliel/OneDrive/Bureau/GAN/%s' % self.dataset_name, exist_ok=True)
        r, c = 3, 3

        #imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True)
        
        imgs_A = self.data_loader.load_data("A", 3, True)
        imgs_B = self.data_loader.load_data("B", 3, True)
        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("C:/Users/eliel/OneDrive/Bureau/GAN/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i), dpi=300)
        plt.close()

        
skillboyz = Pix2Pix()
skillboyz.train(30)
