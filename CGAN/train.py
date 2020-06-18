import sys
import os
from PIL import ImageFile
import datetime

ImageFile.LOAD_TRUNCATED_IMAGES = True

from custom_layers import *
from reseaux import *
from data_loader import DataLoader
from utils import *

#
# Constantes sur l'entrainement
# 

START_EPO = 0
if len(sys.argv) > 1:
    START_EPO = int(sys.argv[1])

BATCH_SIZE = 1 #FIXE ICI, SINON ERREUR DE CALCULS AVEC LE MULTIPLY
EPOCHS = 200
SAMPLE_INTERVAL = 1000

IMG_ROWS = 128
IMG_COLS = 128
CHANNELS = 3
IMG_SHAPE = (IMG_ROWS, IMG_COLS, CHANNELS)

# Configure data loader
dataset_name = 'ugatit'
wf = "Weights/{}/".format(dataset_name)
data_loader = DataLoader(dataset_name=dataset_name, img_res=(IMG_ROWS, IMG_COLS))

# Build and compile the discriminators
d_A, aux_d_A, heatmap_d_A = build_discriminator(IMG_SHAPE, name= "A")
d_B, aux_d_B, heatmap_d_B = build_discriminator(IMG_SHAPE, name= "B")

# Build the generators
g_AB, aux_g_AB, heatmap_g_AB = build_generator(IMG_SHAPE, name= "AB")
g_BA, aux_g_BA, heatmap_g_BA = build_generator(IMG_SHAPE, name= "BA")

# Load
Load_Weights(wf, d_A, d_B, g_AB, g_BA, aux_d_A, aux_d_B, aux_g_AB, aux_g_BA)

# Build the combined model to train generators
combined = build_combined(IMG_SHAPE, d_A, d_B, g_AB, g_BA, aux_d_A, aux_d_B, aux_g_AB, aux_g_BA)


#
# Lancement de l'entrainement 
#

start_time = datetime.datetime.now()

# Adversarial loss ground truths
valid = np.ones((BATCH_SIZE,) + d_A.output_shape[1:])
fake = np.zeros((BATCH_SIZE,) + d_A.output_shape[1:])

aux_valid = np.ones((BATCH_SIZE,) + aux_d_A.output_shape[1:])
aux_fake = np.zeros((BATCH_SIZE,) + aux_d_A.output_shape[1:])

for epoch in range(START_EPO,EPOCHS):
    for batch_i, (imgs_A, imgs_B) in enumerate(data_loader.load_batch(BATCH_SIZE)):

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
            sample_images(data_loader, epoch, batch_i, dataset_name, g_AB, g_BA, heatmap_g_AB, heatmap_g_BA, heatmap_d_A, heatmap_d_B)
            save(wf, d_A, d_B, g_AB, g_BA, aux_d_A, aux_d_B, aux_g_AB, aux_g_BA)