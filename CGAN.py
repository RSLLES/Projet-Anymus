import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from os.path import isfile
from os import remove

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from os import listdir
from time import time
from tqdm import tqdm


########################################
########## Gestion des images ##########
########################################

def compress_images():
    """
    Cette fonction prend toutes les images en .jpg, les import, les transformes en tableau, puis stock le gros tableau résultant
    pour qu'il soit plus simple à charger la prochaine fois
    """
    mainPath = "Horse2Zebra/"

    trainA = load_images(mainPath + "trainA/")
    testA = load_images(mainPath + "testA/")
    dataA = np.concatenate((trainA, testA))

    trainB = load_images(mainPath + "trainB/")
    testB = load_images(mainPath + "testB/")
    dataB = np.concatenate((trainB, testB))

    np.savez_compressed(mainPath + "h2z.npz", dataA, dataB)
    print("dataset saved as h2z.npz" )

def load_compressed_images():
    """
    Charge les images depuis la version compressée fabriquée plus haut
    """
    data = np.load('Horse2Zebra/h2z.npz')
    print("{} loaded".format(str(data['arr_0'].shape)))
    print("{} loaded".format(str(data['arr_1'].shape)))
    return (data['arr_0'], data['arr_1'])

def load_data():
    """Charge les images et les convertis entre -1 et 1 pour être bien utilisée dans la suite """
    XA,XB = load_compressed_images()
    XA = XA/127.5-1
    XB = XB/127.5-1
    return XA,XB

def show_images(dataA, dataB, titleA = [], titleB = []):
    # plot source images
    for i in range(dataA.shape[0]):
        if len(titleA) == 0:
            plt.subplot(2, dataA.shape[0], 1 + i)
        else:
            plt.subplot(2, dataA.shape[0], 1 + i, title=str(titleA[i]))
        plt.axis('off')
        plt.imshow(dataA[i].astype('uint8'))
    # plot target image
    for i in range(dataB.shape[0]):
        if len(titleB) == 0:
            plt.subplot(2, dataB.shape[0], 1 + dataA.shape[0] + i)
        else:
            plt.subplot(2, dataB.shape[0], 1 + dataA.shape[0] + i,title=str(titleB[i]))
        
        plt.axis('off')
        plt.imshow(dataB[i].astype('uint8'))
    plt.show()

def save_images(dataA, dataB, filename):
    # plot source images
    for i in range(dataA.shape[0]):
        plt.subplot(2, dataA.shape[0], 1 + i)
        plt.axis('off')
        plt.imshow(dataA[i].astype('uint8'))
    # plot target image
    for i in range(dataB.shape[0]):
        plt.subplot(2, dataB.shape[0], 1 + dataA.shape[0] + i)
        plt.axis('off')
        plt.imshow(dataB[i].astype('uint8'))
    if isfile(filename):
        remove(filename)
    plt.savefig(filename)


def load_images(path, size=(256,256)):
    """
    Charge les images d'un dossier
    """
    data_list = list()
    for filename in listdir(path):
        pixels = load_img(path + filename, target_size=size)
        pixels = img_to_array(pixels)
        data_list.append(pixels)
    return np.asarray(data_list)
 
#compress_images()
#dataA, dataB = load_compressed_images()
#save_images(dataA[[0,1,2],...], dataB[[0,1,2], ...])



########################################
########## Création du réseau ##########
########################################

def create_discriminator(dim = 256, depht = 32, name=""):
    """
    Retourne un nouveau discriminator type de ceux que nous allons utiliser
    """
    D = keras.models.Sequential(name="d_{}".format(name))
    #Layer 1 : Convolution avec un filtre de 4x4 qui se déplace de 2 pixels en 2 -> Division du nombre de pixel par 2; depht filtres utilisés
    #On ajoute un BatchNormalization pour réduire les poids et éviter une explosion du gradient
    #1] Conv; dim*dim*3 -> dim/2*dim/2*depht
    D.add(keras.layers.Conv2D(depht, (4,4), strides=(2,2), padding="same", activation="relu", input_shape=(dim,dim,3)))
    D.add(keras.layers.BatchNormalization())

    #2] Conv; dim/2*dim/2*depht -> dim/4*dim/4*2*depht
    D.add(keras.layers.Conv2D(2*depht, (4,4), strides=(2,2), padding="same", activation="relu"))
    D.add(keras.layers.BatchNormalization())

    #3] Conv; dim/4*dim/4*2*depht -> dim/8*dim/8*4*depht
    D.add(keras.layers.Conv2D(4*depht, (4,4), strides=(2,2), padding="same", activation="relu"))
    D.add(keras.layers.BatchNormalization())

    #4] Conv; dim/8*dim/8*4*depht -> dim/16*dim/16*8*depht
    D.add(keras.layers.Conv2D(8*depht, (4,4), strides=(2,2), padding="same", activation="relu"))
    D.add(keras.layers.BatchNormalization())

    #5] Conv; dim/16*dim/16*8*depht -> dim/16*dim/16*8*depht
    D.add(keras.layers.Conv2D(8*depht, (4,4), strides=(1,1), padding="same", activation="relu"))
    D.add(keras.layers.BatchNormalization())

    #6] Dense; dim/16*dim/16*8*depht -> 1
    D.add(keras.layers.Flatten())
    D.add(keras.layers.Dense(1, activation="sigmoid"))

    #On compile
    D.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
    return D

def create_generator(dim = 256,depht = 32, n_resnet = 9, name=""):
    """Créer un générateur typique utilisé dans la suite
    La structure est plus complexe, on part de la taille d'une image que l'on compress, puis on réétend pour refaire une image.
    Juste pour essayer, j'essaye aussi une autre notation pour créer des réseaux"""
    input_layer = keras.layers.Input(shape=(dim,dim,3))

    #1] Conv; dim*dim*3 -> dim/2*dim/2*depht
    g = keras.layers.Conv2D(depht, (7,7), strides=(2,2), padding="same")(input_layer)
    g = keras.layers.BatchNormalization()(g)
    g = keras.layers.Activation("relu")(g)

    #2] Conv; dim/2*dim/2*depht -> dim/4*dim/4*2*depht
    g = keras.layers.Conv2D(2*depht, (3,3), strides=(2,2), padding="same")(g)
    g = keras.layers.BatchNormalization()(g)
    g = keras.layers.Activation("relu")(g)

    #3] Conv; dim/4*dim/4*2*depht -> dim/8*dim/8*4*depht
    g = keras.layers.Conv2D(4*depht, (3,3), strides=(2,2), padding="same")(g)
    g = keras.layers.BatchNormalization()(g)
    g = keras.layers.Activation("relu")(g)

    #Au milieu, on ajoute autant de resnet_block que l'on vezut
    for _ in range(n_resnet):
        g = create_resnet(n_filters=4*depht, T = g)
    
    #On redéroule dans l'autre sens
    #4] ConvT; dim/8*dim/8*4*depht -> dim/4*dim/4*2*depht
    g = keras.layers.Conv2DTranspose(2*depht, (3,3), strides=(2,2), padding="same")(g)
    g = keras.layers.BatchNormalization()(g)
    g = keras.layers.Activation("relu")(g)

    #5] ConvT; dim/4*dim/4*2*depht -> dim/2*dim/2*depht
    g = keras.layers.Conv2DTranspose(depht, (3,3), strides=(2,2), padding="same", activation="relu")(g)
    g = keras.layers.BatchNormalization()(g)
    g = keras.layers.Activation("relu")(g)

    #6] ConvT; dim/2*dim/2*depht -> dim*dim*3
    g = keras.layers.Conv2DTranspose(3, (7,7), strides=(2,2), padding="same")(g)
    g = keras.layers.Activation("tanh")(g)
    M = keras.Model(input_layer, g, name="gen_{}".format(name))
    return M


def create_resnet(n_filters, T):
    N = keras.layers.Conv2D(n_filters, (3,3), strides=(1,1), padding="same")(T)
    N = keras.layers.BatchNormalization()(N)
    N = keras.layers.Activation("relu")(N)

    N = keras.layers.Conv2D(n_filters, (3,3), strides=(1,1), padding="same")(N)
    N = keras.layers.BatchNormalization()(N)

    #On additionne l'entrée et la sortie (ce qui fait la particularité du RESNET)
    N = keras.layers.Add()([N, T])

    #Dernière fonction d'activation
    N = keras.layers.Activation("relu")(N)
    return N



############################################################
########## Création des structures d'entrainement ##########
############################################################

def create_training_model_gen(gen_A, d_A, gen_B, dim = 256, name=""):
    """
    Cette méthode combine les différents réseau pour en déduire des fonctions de loss que nous allons chercher a minimiser
    Ici, c'est seulement gen_A qui va être minimisé
    """

    #Entrainement 1 : On veut que gen_a soit capable de tromper le discriminateur
    #On entraine donc le reseau gen_a -> d_A en cherchant à obtenir 1 à chaque fois
    input_from_B = keras.layers.Input(shape=(dim,dim,3))

    gen_A_Out = gen_A(input_from_B)
    d_A_Out = d_A(gen_A_Out)

    #Entrainement 2 et 3 : On veut entrainer le gen_A pour que gen_A = gen_B^-1
    #Cela se traduit par minimisé gen_A(gen_B) et gen_B(gen_A) avec comme objectif l'imagine initiale
    out_direct = gen_B(gen_A_Out)

    input_from_A = keras.layers.Input(shape=(dim,dim,3))
    gen_B_Out = gen_B(input_from_A)
    out_indirect = gen_A(gen_B_Out)

    #Entrainement 4 : Enfin, comme gen_A est sensé transformer une image du monde B vers le monde A,
    #on veut que si on lui donne deja une image du monde A, alors gen_A(A) = A
    out_identity = gen_A(input_from_A)

    #Ces 4 entrainements sont mis ensemble pour etre tous traités en même temps
    model = keras.Model([input_from_A, input_from_B], [d_A_Out, out_direct, out_indirect, out_identity],name="train_gen_{}".format(name))
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    #On prend comme fonction de loss la somme pondéré des 4 fonctions de loss, en donnant plus de poids
    #au entrainement cycle direct et indirect, car d'après le papier ce sont les plus performants
    #mse veut dire mean square error -> c'est la norme 2 qui est utilisé comme loss function pour l'entrainement
    #sur le discriminateur qui vaut 1 (donc on veut minimiser sum (d_A_out-1)^2)
    #mae veut dire mean absolute error -> c'est la norme 1, donc on vise a minimiser sum |y_k-x_k| pour chaque pixel de l'image
    #ce choix vient d'un REX du papier
    model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 2, 2, 1], optimizer=opt)
    return model


##################################
########## Entrainement ##########
##################################
def get_random_element(X, n):
    return X[np.random.randint(0,X.shape[0], n),...]


def train(  gen_A, d_A, gen_B, d_B, 
            training_model_gen_A, training_model_gen_B,  
            XA, XB):

    """C'est ici que se passe le gros entrainement"""
    
    #Caractéristiques de l'entrainement
    n_epochs, n_batch, N_data = 1000, 3, min(XA.shape[0], XB.shape[0])
    n_batch_by_epochs = int(N_data/n_batch)

    #On desactive tout au début de l'entrainement
    gen_A.trainable, gen_B.trainable, d_A.trainable, d_B.trainable = False, False, False, False

    #Et la boucle tourne a tournée

    for i_epo in range(n_epochs):
        print("#######################################")
        print("######## Début epoch {}/{} ############".format(i_epo, n_epochs))
        print("#######################################")

        for i in tqdm(range(n_batch_by_epochs)):
            #Construction du jeu de données a utiliser pour cette iteration de l'entrainement
            xa_real, ya_real = get_random_element(XA, n_batch), np.ones(n_batch)
            xb_real, yb_real = get_random_element(XB, n_batch), np.ones(n_batch)

            xa_fake, ya_fake = gen_A.predict(xb_real), np.zeros(n_batch)
            xb_fake, yb_fake = gen_A.predict(xa_real), np.zeros(n_batch)

            #Entrainements
            #1) On entraine gen_a : [input_from_A, input_from_B] -> [d_A_Out, out_direct, out_indirect, out_identity]
            gen_A.trainable = True
            loss_gen_A = training_model_gen_A.train_on_batch([xa_real, xb_real], [ya_real, xb_real, xa_real, xa_real])
            gen_A.trainable = False

            #2) On entraine gen_b : [input_from_B, input_from_A] -> [d_B_Out, out_direct, out_indirect, out_identity]
            gen_B.trainable = True
            loss_gen_B= training_model_gen_B.train_on_batch([xb_real, xa_real], [yb_real, xa_real, xb_real, xb_real])
            gen_B.trainable = False

            #3) On entraine d_a : input_from_A -> y
            #On l'entraine a la fois avec des vrais données et des fausses
            trainAx, trainAY = np.concatenate((xa_real, xa_fake)), np.concatenate((ya_real, ya_fake))
            d_A.trainable = True
            loss_d_A, acc_d_A = d_A.train_on_batch(trainAx, trainAY)
            d_A.trainable = False

            #4) de même pour d_B
            trainBx, trainBY = np.concatenate((xb_real, xb_fake)), np.concatenate((yb_real, yb_fake))
            d_B.trainable = True
            loss_d_B, acc_d_B = d_B.train_on_batch(trainBx, trainBY)
            d_B.trainable = False

            

        #On affiche un petit résumé de la ou on en est lorsque l'epochs est fini
        print("loss_gen_A ({}): {}".format(training_model_gen_A.metrics_names,loss_gen_A))
        print("loss_gen_B ({}): {}".format(training_model_gen_B.metrics_names,loss_gen_B))
        print("loss_d_A : {} | acc_d_A : {}".format(loss_d_A, acc_d_A))
        print("loss_d_B : {} | acc_d_B : {}".format(loss_d_B, acc_d_B))
        screenshoot(XA, gen_B, i_epo)
        #On lache notre meilleure sauvegarde
        save(d_A, d_B, gen_A, gen_B)


def screenshoot(X, gen, epoch):
    """Fait quelques tests et enregistre l'image pour voir la progression"""
    data1 = (X[[0,1,2],...]+1)*127.5
    data2 = (gen.predict(X[[0,1,2],...])+1)*127.5
    save_images(data1, data2, "Progression/epoch{}.png".format(epoch))

def show_result_network(X):
    data = (X+1)*127.5
    show_images(data, np.array([]))

def test(img, gen, dcorrect, dautre):
    #On va créer les sous titres
    #Pour les img originales du monde correct
    titresimg, titrestransf = [],[]
    predd1, predd2 = dcorrect(img), dautre(img)
    for i in range(img.shape[0]):
        titresimg.append("D1 = {} | D2 = {}".format(predd1[i], predd2[i]))
    predd1, predd2 = dcorrect(gen.predict(img)), dautre(gen.predict(img))
    for i in range(img.shape[0]):
        titrestransf.append("D1 = {} | D2 = {}".format(predd1[i], predd2[i]))


    show_images(img*127.5+127.5, gen.predict(img)*127.5+127.5, titresimg, titrestransf)

def save(d_A, d_B, gen_A, gen_B):
    """Sauvegarde les poids deja calculés, pour pouvoir reprendre les calculs plus tard si jamais"""
    d_A.save_weights("Weights/d_A.h5")
    d_B.save_weights("Weights/d_B.h5")
    gen_A.save_weights("Weights/gen_A.h5")
    gen_B.save_weights("Weights/gen_B.h5")

def load(d_A, d_B, gen_A, gen_B):
    """Sauvegarde les poids deja calculés, pour pouvoir reprendre les calculs plus tard si jamais"""
    if (isfile("Weights/d_A.h5") and isfile("Weights/gen_A.h5") 
    and isfile("Weights/d_B.h5") and isfile("Weights/gen_B.h5")):
        d_A.load_weights("Weights/d_A.h5")
        d_B.load_weights("Weights/d_B.h5")
        gen_A.load_weights("Weights/gen_A.h5")
        gen_B.load_weights("Weights/gen_B.h5")
        print("Weights loaded")
    else:
        print("Missing weights files detected. Starting from scratch")


##################################
########## Let's go baby #########
##################################

def start_train():
    training_model_gen_A, training_model_gen_B = create_training_model_gen(gen_A, d_A, gen_B, name="A"), create_training_model_gen(gen_B, d_B, gen_A, name="B")
    #Et on y va
    train(  gen_A, d_A, gen_B, d_B, training_model_gen_A, training_model_gen_B,  XA, XB)

def essais():
    img = XB[(0,5,10),...]
    test(img, gen_A, d_B, d_A)

dim = 256
XA,XB = load_data()

#Création des discriminateur qui sont eux deja compilés
d_A, d_B = create_discriminator(name="A"), create_discriminator(name="B")

#Au tours des generateurs
gen_A, gen_B = create_generator(name="A"),create_generator(name="B")

#On charge les poids
load(d_A, d_B, gen_A, gen_B)

#essais()
start_train()