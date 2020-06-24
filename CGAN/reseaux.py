from tensorflow.keras.optimizers import Adam
from keras.layers import Input, Concatenate, Lambda, Flatten, Dense, Activation
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.activations import sigmoid
import keras.backend as K
from keras.models import Model

from custom_layers import *

#
# Constantes
#

# Number of filters in the first layer of G and D
GF, DF = 64,64
N_RESNET = 4

# Loss weights
LAMBDA_AUX = 1000
LAMBDA_AUX_DISC = 10
LAMBDA_CYCLE = 10
LAMBDA_ID = 10    

#Optimize
learning_rate = 0.0002
OPTIMIZER = Adam(learning_rate, 0.5)


#
# Generateur et discriminateur
#


def build_generator(IMG_SHAPE, name=""):
    # Image input
    entree = Input(shape=IMG_SHAPE)

    # Downsampling
    g = conv2d(entree, GF, f_size=7, strides=1)
    g = conv2d(g, GF*2, f_size=3, strides=2)
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
    cam = Activation(sigmoid)(cam)
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
    g = deconv2d_adalin(g, GF*2, f_size=4)
    g = deconv2d_adalin(g, GF, f_size=4)

    # Fin du réseau
    g = Conv2D(IMG_SHAPE[-1], kernel_size=4, strides=1, padding='same', activation='tanh')(g)

    g_model = Model(entree, g, name="gen_{}".format(name))
    aux_model = Model(entree, cam, name="aux_gen_{}".format(name))
    heatmap_model = Model(entree, heatmap, name="heatmap_{}".format(name))

    return g_model, aux_model, heatmap_model

def build_discriminator(IMG_SHAPE, name=""):
    # Image input
    entree = Input(shape=IMG_SHAPE)

    # Downsampling
    d = conv2d(entree, DF, f_size=7, strides=1)
    d = conv2d(d, DF*2, f_size=3, strides=2)
    d = conv2d(d, DF*2, f_size=3, strides=2)
    d = conv2d(d, DF*4, f_size=3, strides=2)

    # Création de la CLASSE Activation Map (CAM) en Max et en Average
    d_mp, d_ap = GlobalMaxPooling2D()(d), GlobalAveragePooling2D()(d)

    cam_m, d_m = AuxClass()([d_mp, d])
    cam_a, d_a = AuxClass()([d_ap, d])

    cam = Concatenate()([cam_m, cam_a])
    d = Concatenate()([d_m, d_a])

    d = conv2d(d, DF*4, f_size=1, strides=1)
    cam = Activation(sigmoid)(cam)
    heatmap = Lambda(lambda x: K.sum(x, axis=-1))(d)

    # Final
    d = Conv2D(1, kernel_size=4, strides=1, padding='same')(d)

    d_model = Model(entree, [d,cam], name="disc_{}".format(name))
    d_model.compile(loss=['mse', 'binary_crossentropy'], loss_weights=[1, LAMBDA_AUX_DISC], optimizer=OPTIMIZER, metrics=['accuracy'])

    aux_model = Model(entree, cam, name="aux_d_{}".format(name))
    heatmap_model = Model(entree, heatmap, name="heatmap_{}".format(name))

    return d_model, aux_model, heatmap_model

#
# Model combiné
# 


def build_combined(IMG_SHAPE, d_A, d_B, g_AB, g_BA, aux_g_AB, aux_g_BA):
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

    # Discriminators determines validity of translated images
    valid_A, _ = d_A(fake_A)
    valid_B, _ = d_B(fake_B)

    # Combined model trains generators to fool discriminators
    model = Model(inputs=[img_A, img_B], outputs=[  valid_A, valid_B, 
                                                    reconstr_A, reconstr_B, 
                                                    img_A_id, img_B_id,
                                                    aux_A_dans_AB, aux_B_dans_BA,
                                                    aux_A_dans_BA, aux_B_dans_AB])
    model.compile(  loss=[  'mse', 'mse', 
                            'mae', 'mae',
                            'mae', 'mae',
                            'binary_crossentropy', 'binary_crossentropy',
                            'binary_crossentropy', 'binary_crossentropy'],
                    loss_weights=[  1, 1,
                                    LAMBDA_CYCLE, LAMBDA_CYCLE, 
                                    LAMBDA_ID, LAMBDA_ID,
                                    LAMBDA_AUX, LAMBDA_AUX,
                                    LAMBDA_AUX, LAMBDA_AUX ],
        optimizer=OPTIMIZER)

    return model
