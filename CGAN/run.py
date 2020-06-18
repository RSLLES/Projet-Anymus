import glob
import os
import numpy as np
from tqdm import tqdm
import scipy.misc

from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array

from utils import Load_G_AB
from reseaux import build_generator

def main():
    # Paramètres
    input_folder = "inputs/"
    results_folder = "results/"
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)
    IMG_SHAPE = (128,128,3)

    # On va check le contenu du dossier inputs
    os.makedirs(input_folder, exist_ok=True)
    files = glob.glob(os.path.join(input_folder, "*.jpg"))
    if len(files) == 0:
        print("ERREUR : Il faut au moins une image (format .jpg) à convertir dans le dossier '{}'".format(input_folder))
        return

    # On charge l'architecture
    g, _, _ = build_generator(IMG_SHAPE, name= "AB")
    if not Load_G_AB("Weights/", g):
        return

    # On traite chaque image
    for path in tqdm(files):
        name = os.path.basename(path)
        # Chargement
        pixels = load_img(path, target_size=IMG_SHAPE)
        img = img_to_array(pixels)
        img = img/127.5 - 1.
        img = img[np.newaxis, :, :, :]

        #Conversion
        img_res = g.predict(img)[0,...]
        img_res = 127.5*(img_res+1)

        #Enregistrement
        scipy.misc.imsave(os.path.join(results_folder, name), img_res)

if __name__ == "__main__":
    main()