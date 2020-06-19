import glob
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from MightyMosaic import MightyMosaic

from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array

from utils import Load_G_AB
from reseaux import build_generator

IMG_SHAPE = (128,128,3)
OVERLAP_FACTOR = 4

def load_image(path, shape=None):
    if shape is None:
        pixels = load_img(path)
    else:
        pixels = load_img(path, target_size=shape)
    img = img_to_array(pixels)
    img = img/127.5 - 1.
    img = img[np.newaxis, :, :, :]
    return img

def run_folder_goodsize():
    # Paramètres
    input_folder = "inputs/"
    results_folder = "results/"
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

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
        img = load_image(path, shape=IMG_SHAPE)

        #Conversion
        img_res = g.predict(img)[0,...]
        img_res = 127.5*(img_res+1)

        #Enregistrement
        pil_img = Image.fromarray(img_res.astype(np.uint8))
        pil_img.save(os.path.join(results_folder, name))

def run_img_bigger(path):
    #Vérification de l'existence du fichier
    if not os.path.isfile(path):
        print("L'adresse '{}' ne pointe vers aucune image.")
        return False

    # On importe l'image
    img = load_image(path)[0,...]
    h,w,c = img.shape
    print("Image size : {}x{}".format(w,h))
    if h%OVERLAP_FACTOR != 0 or w%OVERLAP_FACTOR != 0:
        while h%OVERLAP_FACTOR != 0:
            h+=1
        while w%OVERLAP_FACTOR != 0:
            w+=1
        img = load_image(path, shape=(h,w,c))[0,...]
        print("L'image est redimensionnée en {}x{}".format(w,h))

    # On charge l'architecture
    g, _, _ = build_generator(IMG_SHAPE, name= "AB")
    if not Load_G_AB("Weights/", g):
        return False

    # Transformation en mosaic
    mosaic = MightyMosaic.from_array(img, (IMG_SHAPE[0], IMG_SHAPE[1]), overlap_factor=OVERLAP_FACTOR)
    print(f'The mosaic shape is {mosaic.shape} : {mosaic.shape[0]*mosaic.shape[1]} mosaics to transform')

    # Conversion
    print("Fusion")
    prediction = mosaic.apply(g.predict, progress_bar=True, batch_size=1)
    prediction = prediction.get_fusion()

    # Enregistrement
    img_res = 127.5*(prediction+1)
    pil_img = Image.fromarray(img_res.astype(np.uint8))
    pil_img.save(f"results/{path}")
    print("Success.")

def run_all_folder():
    # Paramètres
    input_folder = "inputs/"
    results_folder = "results/"
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

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
    i=0
    for path in files:
        i+=1
        name = os.path.basename(path)
        print(f"[{i}] {name}")
        # Chargement
        img = load_image(path)[0,...]
        h,w,c = img.shape
        print("Image size : {}x{}".format(w,h))
        if h%OVERLAP_FACTOR != 0 or w%OVERLAP_FACTOR != 0:
            while h%OVERLAP_FACTOR != 0:
                h+=1
            while w%OVERLAP_FACTOR != 0:
                w+=1
            img = load_image(path, shape=(h,w,c))[0,...]
            print("L'image est redimensionnée en {}x{}".format(w,h))

        # Transformation en mosaic
        mosaic = MightyMosaic.from_array(img, (IMG_SHAPE[0], IMG_SHAPE[1]), overlap_factor=OVERLAP_FACTOR)
        print(f'The mosaic shape is {mosaic.shape} : {mosaic.shape[0]*mosaic.shape[1]} mosaics to transform')

        # Conversion
        print("Fusion")
        prediction = mosaic.apply(g.predict, progress_bar=True, batch_size=1)
        prediction = prediction.get_fusion()

        # Enregistrement
        img_res = 127.5*(prediction+1)

        #Enregistrement
        pil_img = Image.fromarray(img_res.astype(np.uint8))
        pil_img.save(os.path.join(results_folder, name))


if __name__ == "__main__":
    run_all_folder()