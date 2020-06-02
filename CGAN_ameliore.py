import keras
import numpy as np
import sys
from os.path import isfile
from os import remove

from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

from keras.utils.vis_utils import plot_model

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

    dataA = load_images("2000Faces/", size=(128,128))
    dataB = load_images("2000Manga/", size=(128,128))

    np.savez_compressed("f2m.npz", dataA, dataB)
    print("dataset saved as f2m.npz" )

def load_compressed_images():
    """
    Charge les images depuis la version compressée fabriquée plus haut
    """
    data = np.load('f2m.npz')
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


def load_images(path, size):
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

def create_discriminator(dim, depht = 32, name=""):
    """
    On change la structure par / à CGAN.py, voir pdf page 6 figure 2
    """
    input_layer = keras.layers.Input(shape=(dim,dim,3))
    #Layer 1 : Convolution avec un filtre de 4x4 qui se déplace de 2 pixels en 2 -> Division du nombre de pixel par 2; depht filtres utilisés
    #On ajoute un InstanceNormalization pour réduire les poids et éviter une explosion du gradient
    #1] Conv; dim*dim*3 -> dim/2*dim/2*2depht
    d = keras.layers.Conv2D(2*depht, (4,4), strides=(2,2), padding="same")(input_layer)
    d = InstanceNormalization(axis=-1)(d)
    d = keras.layers.LeakyReLU(alpha=0.2)(d)

    #2] Conv; dim/2*dim/2*depht -> dim/4*dim/4*4*depht
    d = keras.layers.Conv2D(4*depht, (4,4), strides=(2,2), padding="same")(d)
    d = InstanceNormalization(axis=-1)(d)
    d = keras.layers.LeakyReLU(alpha=0.2)(d)

    #3] Conv; dim/4*dim/4*2*depht -> dim/8*dim/8*8*depht
    d = keras.layers.Conv2D(8*depht, (4,4), strides=(2,2), padding="same")(d)
    d = InstanceNormalization(axis=-1)(d)
    d = keras.layers.LeakyReLU(alpha=0.2)(d)

    #4] Conv; dim/8*dim/8*8*depht -> dim/8*dim/8*8*depht
    d = keras.layers.Conv2D(8*depht, (3,3), strides=(1,1), padding="same")(d)
    d = InstanceNormalization(axis=-1)(d)
    pre_dil_conv = keras.layers.LeakyReLU(alpha=0.2)(d)

    #C'est ici que se trouve un premier skip d'après le papier, on continue donc de l'autre coté avec des reseau de convolutions
    #dilués et l'on ferra une concatenation plus loin pour permettre la connection

    #5] Dil Conv de d = 2
    d = keras.layers.Conv2D(8*depht, (3,3), strides=(1,1), dilation_rate=(2,2),padding="same")(pre_dil_conv)
    d = InstanceNormalization(axis=-1)(d)
    d = keras.layers.LeakyReLU(alpha=0.2)(d)

    #6] Dil Conv de d = 4
    d = keras.layers.Conv2D(8*depht, (3,3), strides=(1,1), dilation_rate=(4,4),padding="same")(d)
    d = InstanceNormalization(axis=-1)(d)
    d = keras.layers.LeakyReLU(alpha=0.2)(d)

    #7] Dil Conv de d = 8
    d = keras.layers.Conv2D(8*depht, (3,3), strides=(1,1), dilation_rate=(8,8),padding="same")(d)
    d = InstanceNormalization(axis=-1)(d)
    post_dil_conv = keras.layers.LeakyReLU(alpha=0.2)(d)

    #8] Reconnection par concatenation
    d = keras.layers.Concatenate()([pre_dil_conv, post_dil_conv])

    #9] Fin du réseau : Conv
    d = keras.layers.Conv2D(8*depht, (3,3), strides=(1,1), padding="same")(d)
    d = InstanceNormalization(axis=-1)(d)
    d = keras.layers.LeakyReLU(alpha=0.2)(d)

    #10] Dernier conv pour avoir un vecteur
    d = keras.layers.Conv2D(1, (4,4), strides=(1,1), padding="same")(d)

    #On compile
    model = keras.Model(input_layer, d)
    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.5)
    model.compile(loss='mse', optimizer=opt, loss_weights=[0.5], metrics=["accuracy"])

    #Enfin, on enregistre dans un fichier si jamais c'est demandé pour vérifier la structure du réseau
    #plot_model(d, to_file="d_{}.png".format(name), show_shapes=True, show_layer_names=True)

    return model

def create_generator(dim, depht = 32, name=""):
    """    On change la structure par / à CGAN.py, voir pdf """
    input_layer = keras.layers.Input(shape=(dim,dim,3))

    #1) Convolution (dim,dim,3) -> (dim/2,dim/2,depht)
    g = keras.layers.Conv2D(depht, (4,4), strides=(2,2), padding="same")(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = keras.layers.LeakyReLU(alpha=0.2)(g)

    #2) Convolution (dim/2,dim/2,depht) -> (dim/2,dim/2,4*depht)
    g = keras.layers.Conv2D(4*depht, (4,4), strides=(1,1), padding="same")(g)
    g = InstanceNormalization(axis=-1)(g)
    g = keras.layers.LeakyReLU(alpha=0.2)(g)

    #3) 3 RESNET : (dim/2,dim/2,4*depht)
    g = create_resnet(g)
    g = create_resnet(g)
    point_1 = create_resnet(g)

    #4) Convolution : (dim/2,dim/2,4*depht) -> (dim/4,dim/4,8*depht)
    g = keras.layers.Conv2D(8*depht, (4,4), strides=(2,2), padding="same")(point_1)
    g = InstanceNormalization(axis=-1)(g)
    g = keras.layers.LeakyReLU(alpha=0.2)(g)

    #4) 3 Resnet : (dim/4,dim/4,8*depht)
    g = create_resnet(g)
    g = create_resnet(g)
    point_2 = create_resnet(g)

    #5) Convolution (dim/4,dim/4,8*depht) -> (dim/8,dim/8,8*depht)
    g = keras.layers.Conv2D(8*depht, (4,4), strides=(2,2), padding="same")(point_2)
    g = InstanceNormalization(axis=-1)(g)
    g = keras.layers.LeakyReLU(alpha=0.2)(g)

    #3) RESNET
    for _ in range(3):
        g = create_resnet(g)

    #4) Deconv : (dim/8,dim/8,8*depht) -> (dim/4,dim/4,8*depht)
    g = keras.layers.Conv2DTranspose(8*depht, (3,3), strides=(2,2), padding="same")(g)
    g = InstanceNormalization(axis=-1)(g)
    g = keras.layers.LeakyReLU(alpha=0.2)(g)

    #Raccord 2 : (dim/4,dim/4,8*depht) + (dim/4,dim/4,8*depht) -> (dim/4,dim/4,16*depht)
    g = keras.layers.Concatenate()([g, point_2])

    #3 RESNET (dim/4,dim/4,16*depht)
    for _ in range(3):
        g = create_resnet(g)

    #Transpose Conv : (dim/4,dim/4,16*depht) -> (dim/2,dim/2,4*depht)
    g = keras.layers.Conv2DTranspose(4*depht, (4,4), strides=(2,2), padding="same")(g)
    g = InstanceNormalization(axis=-1)(g)
    g = keras.layers.LeakyReLU(alpha=0.2)(g)

    #Raccord 1 : (dim/2,dim/2,4*depht) + (dim/2,dim/2,4*depht) -> (dim/2,dim/2,8*depht)
    g = keras.layers.Concatenate()([g, point_1])

    #3) RESNET (dim/2,dim/2,8*depht)
    for _ in range(3):
        g = create_resnet(g)

    #DeConvolution : (dim/2,dim/2,8*depht) -> (dim,dim,depht)
    g = keras.layers.Conv2DTranspose(depht, (4,4), strides=(2,2), padding="same")(g)
    g = InstanceNormalization(axis=-1)(g)
    g = keras.layers.LeakyReLU(alpha=0.2)(g)

    #DeConvolution : (dim,dim,depht) -> (dim,dim,3)
    g = keras.layers.Conv2DTranspose(3, (4,4), strides=(1,1), padding="same")(g)
    g = keras.layers.Activation("tanh")(g)

    M = keras.Model(input_layer, g, name="gen_{}".format(name))
    return M


def create_resnet(T):
    n_filters = T.shape[-1]
    N = keras.layers.Conv2D(n_filters, (3,3), strides=(1,1), padding="same")(T)
    N = InstanceNormalization(axis=-1)(N)
    N = keras.layers.LeakyReLU(alpha=0.2)(N)

    N = keras.layers.Conv2D(n_filters, (3,3), strides=(1,1), padding="same")(N)
    N = InstanceNormalization(axis=-1)(N)

    #On additionne l'entrée et la sortie (ce qui fait la particularité du RESNET)
    #L'addition fonctionne forcément vu que le nombre de filtre correspond bien et que l'on a pas touché a la taille du reseau
    N = keras.layers.Add()([N, T])

    #Dernière fonction d'activation
    N = keras.layers.LeakyReLU(alpha=0.2)(N)
    return N



############################################################
########## Création des structures d'entrainement ##########
############################################################

def create_training_model_gen(gen_1_vers_2, d_2, gen_2_vers_1, dim, name=""):
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
    model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 10, 10, 5], optimizer=opt, metrics=["accuracy"])
    return model


##################################
########## Entrainement ##########
##################################
def get_random_element(X, n):
    return X[np.random.randint(0,X.shape[0], n),...]

def update_pool(existing_pool, new_images, pool_max_size=50):
    """
    Cette fonction tient un historique des dernières images générée, car d'après le document il est plus performant
    d'entrener le générateur sur des images précédemment générées."""
    selected = []
    for img in new_images:
        #Si le pool n'est pas encore rempli, on l'ajoute au pool
        if (len(existing_pool) < pool_max_size):
            existing_pool.append(img)
            selected.append(img)
        #Sinon une chance sur deux de l'utilisé, ou bien d'utilisé une vielle image
        elif np.random.random() > 0.5 :
            #On l'utilise mais on ne l'ajoute pas dans le pool
            selected.append(img)
        else:
            #On utilise une ancienne image que l'on enleve du pool
            index = np.random.randint(0,len(existing_pool))
            selected.append(existing_pool[index])
            existing_pool[index] = img
    return np.asarray(selected)


def train(  gen_A_vers_B, d_A, gen_B_vers_A, d_B, 
            training_model_gen_A_vers_B, training_model_gen_B_vers_A,  
            XA, XB,
            starting_epoch = 0):

    """C'est ici que se passe le gros entrainement"""
    
    #Caractéristiques de l'entrainement
    n_epochs, n_batch, N_data, period_screen = 1000, 3, max(XA.shape[0], XB.shape[0]), 1
    d_update_period = 3
    n_run_by_epochs = int(N_data/n_batch)
    shape_y = (n_batch, d_A.output_shape[1], d_A.output_shape[2], d_A.output_shape[3])

    #Et la boucle qui tourne a tournée (ty Ribery)
    for i_epo in range(starting_epoch,n_epochs):
        print("")
        print("#######################################")
        print("######## Début epoch {}/{} ############".format(i_epo, n_epochs))
        print("#######################################")

        loss_gen_A_vers_B, loss_gen_B_vers_A = [],[]
        loss_d_A, loss_d_B = [],[]
        poolA, poolB = [],[]

        for i in tqdm(range(n_run_by_epochs)):
            #Construction du jeu de données a utiliser pour cette iteration de l'entrainement
            xa_real, ya_real = get_random_element(XA, n_batch), np.ones(shape_y).astype(np.float32)
            xb_real, yb_real = get_random_element(XB, n_batch), np.ones(shape_y).astype(np.float32)

            xa_fake, ya_fake = update_pool(poolA, gen_B_vers_A.predict(xb_real)), np.zeros(shape_y).astype(np.float32)
            xb_fake, yb_fake = update_pool(poolB, gen_A_vers_B.predict(xa_real)), np.zeros(shape_y).astype(np.float32)

            #Entrainements
            #1) On entraine gen_A_vers_B : ici, le monde 1 est A et le monde 2 est B
            #on avait gen_1_vers_2 : [input_from_1, input_from_2] -> [pred_d2, cycle_1, cycle_2, identity_2]
            e1 = training_model_gen_A_vers_B.train_on_batch([xa_real, xb_real], [yb_real, xa_real, xb_real, xb_real])
            loss_gen_A_vers_B.append(np.array(e1))

            #4) de même pour d_B
            #train_discriminator_with_threshold(d_B, xb_real, xb_fake, yb_real, yb_fake, loss_d_B)
            train_discriminator_with_period(d_B, xb_real, xb_fake, yb_real, yb_fake, i, d_update_period)

            #2) Sur le meme model, on entraine gen_B_vers_A
            # gen_1_vers_2 : [input_from_1, input_from_2] -> [pred_d2, cycle_1, cycle_2, identity_2]
            e2 = training_model_gen_B_vers_A.train_on_batch([xb_real, xa_real], [ya_real, xb_real, xa_real, xa_real])
            loss_gen_B_vers_A.append(np.array(e2))

            #3) On entraine d_A : input_from_A -> y
            #On l'entraine a la fois avec des vrais données et des fausses
            #On l'entraine uniquement si il n'est pas deja trop fort, donc si sa précision ne dépasse pas le treshold indiqué
            #train_discriminator_with_threshold(d_A, xa_real, xa_fake, ya_real, ya_fake, loss_d_A)
            train_discriminator_with_period(d_A, xa_real, xa_fake, ya_real, ya_fake, loss_d_B, i, d_update_period)

        #On affiche un petit résumé de la ou on en est lorsque l'epochs est fini
        #Calcul des moyennes au cours de l'epoch
        avg_loss_gen_A_vers_B = sum(loss_gen_A_vers_B)/n_run_by_epochs
        avg_loss_gen_B_vers_A = sum(loss_gen_B_vers_A)/n_run_by_epochs
        avg_loss_d_A = sum(loss_d_A)/n_run_by_epochs
        avg_loss_d_B = sum(loss_d_B)/n_run_by_epochs


        print("Bilan de l'epoch :")
        print("loss gen_A_vers_B : {}".format(loss_info(avg_loss_gen_A_vers_B)))
        print("loss gen_B_vers_A : {}".format(loss_info(avg_loss_gen_B_vers_A)))
        print("loss d_A : {}".format(loss_info(avg_loss_d_A)))
        print("loss d_B : {}".format(loss_info(avg_loss_d_B)))

        #Toutes les period_screen epochs, on fait un sourire
        if (i_epo)%period_screen == 0:
            screenshoot(XA, gen_A_vers_B, str(i_epo) + "_A_vers_B")
            screenshoot(XB, gen_B_vers_A, str(i_epo) + "_B_vers_A")
        
        #On lache notre meilleure sauvegarde
        save(d_A, d_B, gen_A_vers_B, gen_B_vers_A)

def loss_info (loss) : 
    return [str(loss[i]) for i in range(loss.shape[0])]

def train_discriminator_with_threshold(d, x_real, x_fake, y_real, y_fake, loss, d_accuracy_threshold = 0.80):
    x, y = np.concatenate((x_real, x_fake)), np.concatenate((y_real, y_fake))
    e = d.test_on_batch(x, y)
    if (len(loss) == 0 or e[1] <= d_accuracy_threshold):
        e = d_B.train_on_batch(x, y)
    loss.append(np.array(e))

def train_discriminator_with_period(d, x_real, x_fake, y_real, y_fake, loss, i, period = 2):
    if i%period == 0:
        x, y = np.concatenate((x_real, x_fake)), np.concatenate((y_real, y_fake))
        e = d_B.train_on_batch(x, y)
        loss.append(np.array(e))

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

dim = 128
XA,XB = load_data()

#Création des discriminateur qui sont eux deja compilés
d_A, d_B = create_discriminator(dim, name="A"), create_discriminator(dim, name="B")

#Au tours des generateurs
gen_A_vers_B, gen_B_vers_A = create_generator(dim, name="A_vers_B"), create_generator(dim, name="B_vers_A")

#On charge les poids
load(d_A, d_B, gen_A_vers_B, gen_B_vers_A)

#On creer les training model
#gen_1_vers_2 : create_training_model_gen(gen_1_vers_2, d_2, gen_2_vers_1, name="")
#swapped
training_model_gen_B_vers_A = create_training_model_gen(gen_B_vers_A, d_A, gen_A_vers_B, dim, name="B_vers_A")
training_model_gen_A_vers_B = create_training_model_gen(gen_A_vers_B, d_B, gen_B_vers_A, dim, name="A_vers_B")

#Et on y va
starting_epoch = 0
if (len(sys.argv) > 1):
    starting_epoch = int(sys.argv[1])

train(gen_A_vers_B, d_A, gen_B_vers_A, d_B, training_model_gen_A_vers_B, training_model_gen_B_vers_A,  XA, XB, starting_epoch)