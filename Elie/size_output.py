# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:19:17 2020

@author: eliel
"""
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

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)


        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([img_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, img_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)
        
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