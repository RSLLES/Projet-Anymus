# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:19:17 2020

@author: eliel
"""

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
    d7 = d_layer(d6, 1, f_size = 3, l_stride = 1, is_norm = False, is_relu = False)

    validity = d7

    return Model([img_A, img_B], validity)