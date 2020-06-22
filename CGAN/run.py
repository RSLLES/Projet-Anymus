IMG_SHAPE = (256,256,3)
OVERLAP_FACTOR = 4

def load_model(path):
    g,_,_ = build_generator(IMG_SHAPE, name="")
    Load_G_AB(path, g)
    return g


def load_image(path, shape=None):
    if shape is None:
        pixels = load_img(path)
    else:
        pixels = load_img(path, target_size=shape)
    img = img_to_array(pixels)
    img = img/127.5 - 1.
    img = img[np.newaxis, :, :, :]
    return img

def run_goodsize(path, path_to_gen, preffix, results_folder):
    os.makedirs(results_folder, exist_ok=True)

    # On charge l'architecture
    g = load_model(path_to_gen)

    name = os.path.basename(path)
    # Chargement
    img = load_image(path, shape=IMG_SHAPE)

    #Conversion
    img_res = g.predict(img)[0,...]
    img_res = 127.5*(img_res+1)

    #Enregistrement
    pil_img = Image.fromarray(img_res.astype(np.uint8))
    pil_img.save(os.path.join(results_folder, f"{preffix}-{name}"))

def run_folder_goodsize(input_folder, path_to_gen, preffix, results_folder):
    # Paramètres
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    # On va check le contenu du dossier inputs
    os.makedirs(input_folder, exist_ok=True)
    files = glob.glob(os.path.join(input_folder, "*.jpg"))
    if len(files) == 0:
        print("ERREUR : Il faut au moins une image (format .jpg) à convertir dans le dossier '{}'".format(input_folder))
        return

    # On charge l'architecture
    g = load_model(path_to_gen)

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
        pil_img.save(os.path.join(results_folder, f"{preffix}-{name}"))

def run_img_bigger(path, path_to_gen, preffix, results_folder):
    os.makedirs(results_folder, exist_ok=True)
    # On importe l'image
    name = os.path.basename(path)
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
    g = load_model(path_to_gen)

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
    pil_img.save(os.path.join(results_folder, f"{preffix}-{name}"))
    print("Success.")

def run_all_folder_bigger(input_folder, path_to_gen, preffix, results_folder):
    # Paramètres
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    # On va check le contenu du dossier inputs
    os.makedirs(input_folder, exist_ok=True)
    files = glob.glob(os.path.join(input_folder, "*.jpg")) + glob.glob(os.path.join(input_folder, "*.png"))
    if len(files) == 0:
        print("ERREUR : Il faut au moins une image (d'extension .jpg ou .png) à convertir dans le dossier '{}'".format(input_folder))
        return

    # On charge l'architecture
    g = load_model(path_to_gen)

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
        pil_img.save(os.path.join(results_folder, f"{preffix}-{name}"))


if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser(description='Permet de faire tourner le réseau de transformation Visage -> Manga sur des images. ')
    parser.add_argument('image', help="Chemin vers l'image à traiter. Cela peut aussi etre un chemin vers un dossier, auquel cas toutes les images trouvées dedans seront traitées")
    parser.add_argument("-o", default="results/", help="Dossier dans lequel seront stockés les résultats. Par défaut 'results/'")
    parser.add_argument("-g", default="g.h5", help="Chemin vers le fichier contenant les poids du générateur à utiliser. Par défaut 'g.h5'")
    parser.add_argument("-p", default="result", help="Ajoute le prefixe prefixe-nom_de_l'image_original.jpg au résultat. 'result' par défaut.")
    parser.add_argument("-m", "--mosaic", action="store_true", help="Pour traiter des images hd qui ne doivent pas être redimensionnées en 256x256. Elles seront alors composées d'une mosaic de carrés de tailles 256x256.")
    parser.add_argument("-v", "--verbose", help="Affiche les logs de tensorflow", action="store_true")
    args = parser.parse_args()

    #Verbose
    if args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    #Importation
    import glob
    import numpy as np
    from tqdm import tqdm
    from PIL import Image
    from MightyMosaic import MightyMosaic

    from keras_preprocessing.image import load_img
    from keras_preprocessing.image import img_to_array

    from reseaux import build_generator
    from utils import Load_G_AB


    #On vérifie ce que l'on trouve sur le chemin donné
    isFile,isDirectory = os.path.isfile(args.image), os.path.isdir(args.image)

    if not isFile and not isDirectory:
        print(f"ERREUR : Le path '{args.image}' ne mene nul part.'")
    else:
        if isFile and not args.mosaic:
            run_goodsize(args.image, args.g, args.p, args.o)

        if isFile and args.mosaic:
            run_img_bigger(args.image, args.g, args.p, args.o)

        if isDirectory and not args.mosaic:
            run_folder_goodsize(args.image, args.g, args.p, args.o)

        if isDirectory and args.mosaic:
            run_all_folder_bigger(args.image, args.g, args.p, args.o)