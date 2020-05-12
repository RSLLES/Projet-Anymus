import keras
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

print(keras.__version__)

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
    D.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5), metrics=["accuracy"])
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

def create_small_training_model_gen(gen, d, dim=256):
    input_layer = keras.layers.Input(shape=(dim,dim,3))
    output_layer = d(gen(input_layer))
    model = keras.Model(input_layer, output_layer)
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='mse', optimizer=opt)
    return model

def create_training_model_gen(gen_1_vers_2, d_2, gen_2_vers_1, dim = 256, name=""):
    """
    Cette méthode combine les différents réseau pour en déduire des fonctions de loss que nous allons chercher a minimiser
    Ici, c'est seulement gen_1_vers_2 qui va être entrainé
    """

    #On desactive tout sauf gen_1_vers_2
    gen_1_vers_2.trainable = True
    gen_2_vers_1.trainable = False
    d_2.trainable = False

    #Voila les deux entrees de ce model : une entree du monde 1 et une autre du monde 2
    input_from_1 = keras.layers.Input(shape=(dim,dim,3))
    input_from_2 = keras.layers.Input(shape=(dim,dim,3))

    #Entrainement 1 : On veut que gen_1_vers_2 soit capable de tromper le discriminateur d2
    #On entraine donc le reseau d_2(gen_1_vers_2(input_from_1)) en cherchant à obtenir 1 à chaque fois
    pred_d2 = d_2(gen_1_vers_2(input_from_1))

    #Entrainement 2 et 3 : L'objectif est que logiquement, gen_1_vers_2 = gen_2_vers_1^-1
    #Donc on va s'entrainer sur deux boucles, gen_1_vers_2(gen_2_vers_1(input_from_2)) = input_from_2
    # et dans l'autre sens gen_2_vers_1(gen_1_vers_2(input_1)) = input_1
    cycle_1 = gen_2_vers_1(gen_1_vers_2(input_from_1))
    cycle_2 = gen_1_vers_2(gen_2_vers_1(input_from_2))

    #Entrainement 4 : Enfin, une image du monde 2 ne doit pas changer par gen_1_vers_2
    identity_2 = gen_1_vers_2(input_from_2)

    #Ces 4 entrainements sont mis ensemble pour etre tous traités en même temps
    model = keras.Model([input_from_1, input_from_2], [pred_d2, cycle_1, cycle_2, identity_2],name="train_gen_{}".format(name))
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)

    #Compilation du model, on va minimiser la CL de ces fonctions de pertes, pondéré par les poids en dessous
    # (on donne plus d'importance aux cycles d'après le papier)
    model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 2, 2, 1], optimizer=opt, metrics=["accuracy"])
    return model


##################################
########## Entrainement ##########
##################################
def get_random_element(X, n):
    return X[np.random.randint(0,X.shape[0], n),...]


def train(  gen_A_vers_B, d_A, gen_B_vers_A, d_B, 
            training_model_gen_A_vers_B, training_model_gen_B_vers_A,  
            XA, XB):

    """C'est ici que se passe le gros entrainement"""
    
    #Caractéristiques de l'entrainement
    n_epochs, n_batch, N_data = 1000, 20, max(XA.shape[0], XB.shape[0])
    n_run_by_epochs = int(N_data/n_batch)

    #Et la boucle qui tourne a tournée (ty Ribery)
    for i_epo in range(n_epochs):
        print("#######################################")
        print("######## Début epoch {}/{} ############".format(i_epo, n_epochs))
        print("#######################################")

        loss_gen_A_vers_B, loss_gen_B_vers_A = [],[]
        loss_d_A, loss_d_B = [],[]

        for i in tqdm(range(n_run_by_epochs)):
            #Construction du jeu de données a utiliser pour cette iteration de l'entrainement
            xa_real, ya_real = get_random_element(XA, n_batch), (np.ones(n_batch)[...,np.newaxis]).astype(np.float32)
            xb_real, yb_real = get_random_element(XB, n_batch), (np.ones(n_batch)[...,np.newaxis]).astype(np.float32)

            xa_fake, ya_fake = gen_B_vers_A.predict(xb_real), (np.zeros(n_batch)[...,np.newaxis]).astype(np.float32)
            xb_fake, yb_fake = gen_A_vers_B.predict(xa_real), (np.zeros(n_batch)[...,np.newaxis]).astype(np.float32)

            #Entrainements
            #1) On entraine gen_A_vers_B : ici, le monde 1 est A et le monde 2 est B
            #on avait gen_1_vers_2 : [input_from_1, input_from_2] -> [pred_d2, cycle_1, cycle_2, identity_2]
            e1 = training_model_gen_A_vers_B.train_on_batch([xa_real, xb_real], [yb_real, xa_real, xb_real, xb_real])
            loss_gen_A_vers_B.append(np.array(e1))

            #2) Sur le meme model, on entraine gen_B_vers_A
            # gen_1_vers_2 : [input_from_1, input_from_2] -> [pred_d2, cycle_1, cycle_2, identity_2]
            e2 = loss_gen_B_vers_A = training_model_gen_B_vers_A.train_on_batch([xb_real, xa_real], [ya_real, xb_real, xa_real, xa_real])
            loss_gen_B_vers_A.append(np.array(e2))

            #3) On entraine d_A : input_from_A -> y
            #On l'entraine a la fois avec des vrais données et des fausses
            xa, ya = np.concatenate((xa_real, xa_fake)), np.concatenate((ya_real, ya_fake))
            e3 = d_A.train_on_batch(xa, ya)
            loss_d_A.append(np.array(e3))

            #4) de même pour d_B
            xb, yb = np.concatenate((xb_real, xb_fake)), np.concatenate((yb_real, yb_fake))
            e4 = d_B.train_on_batch(xb, yb)
            loss_d_B.append(np.array(e4))


        #On affiche un petit résumé de la ou on en est lorsque l'epochs est fini
        #Calcul des moyennes au cours de l'epoch
        avg_loss_gen_A_vers_B = sum(loss_gen_A_vers_B)/n_run_by_epochs
        avg_loss_gen_B_vers_A = sum(loss_gen_B_vers_A)/n_run_by_epochs
        avg_loss_d_A = sum(loss_d_A)/n_run_by_epochs
        avg_loss_d_B = sum(loss_d_B)/n_run_by_epochs


        print("Bilan de l'epoch :")
        print("loss gen_A_vers_B : {}".format(loss_info(gen_A_vers_B, avg_loss_gen_A_vers_B)))
        print("loss gen_B_vers_A : {}".format(loss_info(gen_B_vers_A, avg_loss_gen_B_vers_A)))
        print("loss d_A : {}".format(loss_info(d_A, avg_loss_d_A)))
        print("loss d_B : {}".format(loss_info(d_B, avg_loss_d_B)))

        #Toutes les 5 epochs, on fait un sourire
        if (i_epo)%5 == 0:
            screenshoot(XA, gen_A_vers_B, str(i_epo) + "_A_vers_B")
            screenshoot(XB, gen_B_vers_A, str(i_epo) + "_B_vers_A")
        
        #On lache notre meilleure sauvegarde
        save(d_A, d_B, gen_A_vers_B, gen_B_vers_A)

def loss_info (r,loss) : 
    return [str(r.metrics_name[i]) + " : " + str(loss[i]) for i in range(loss.shape[0])]


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

def save(d_A, d_B, gen_A_vers_B, gen_B_vers_A):
    """Sauvegarde les poids deja calculés, pour pouvoir reprendre les calculs plus tard si jamais"""
    d_A.save_weights("Weights/d_A.h5")
    d_B.save_weights("Weights/d_B.h5")
    gen_A_vers_B.save_weights("Weights/gen_A_vers_B.h5")
    gen_B_vers_A.save_weights("Weights/gen_B_vers_A.h5")

def load(d_A, d_B, gen_A_vers_B, gen_B_vers_A):
    """Sauvegarde les poids deja calculés, pour pouvoir reprendre les calculs plus tard si jamais"""
    if (isfile("Weights/d_A.h5") and isfile("Weights/gen_A_vers_B.h5") 
    and isfile("Weights/d_B.h5") and isfile("Weights/gen_B_vers_A.h5")):
        d_A.load_weights("Weights/d_A.h5")
        d_B.load_weights("Weights/d_B.h5")
        gen_A_vers_B.load_weights("Weights/gen_A_vers_B.h5")
        gen_B_vers_A.load_weights("Weights/gen_B_vers_A.h5")
        print("Weights loaded")
    else:
        print("Missing weights files detected. Starting from scratch")




##################################
########## Let's go baby #########
##################################


dim = 256
XA,XB = load_data()

#Création des discriminateur qui sont eux deja compilés
d_A, d_B = create_discriminator(name="A"), create_discriminator(name="B")

#Au tours des generateurs
gen_A_vers_B, gen_B_vers_A = create_generator(name="A_vers_B"), create_generator(name="B_vers_A")

#On charge les poids
load(d_A, d_B, gen_A_vers_B, gen_B_vers_A)

#On creer les training model
#gen_1_vers_2 : create_training_model_gen(gen_1_vers_2, d_2, gen_2_vers_1, dim = 256, name="")
training_model_gen_A_vers_B = create_training_model_gen(gen_A_vers_B, d_B, gen_B_vers_A, name="A_vers_B")
training_model_gen_B_vers_A = create_training_model_gen(gen_B_vers_A, d_A, gen_A_vers_B, name="B_vers_A")

#Et on y va
train(gen_A_vers_B, d_A, gen_B_vers_A, d_B, training_model_gen_A_vers_B, training_model_gen_B_vers_A,  XA, XB)