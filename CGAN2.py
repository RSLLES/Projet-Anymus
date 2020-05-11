#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""import sys
!conda install --yes --prefix {sys.prefix} tensorflow
ou
!{sys.executable} -m pip install numpy"""

import keras
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras_preprocessing.image import load_img, img_to_array
from os import listdir

keras.__version__


# ### 1) Gestion des images

# In[ ]:


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

def compress_images():
    """
    Cette fonction prend toutes les images en .jpg, les import, les transformes en tableau, puis stock le gros tableau résultant
    pour qu'il soit plus simple à charger la prochaine fois
    """
    mainPath = "Horse2Zebra/"

    trainA = load_images(mainPath + "trainA/")
    testA = load_images(mainPath + "testA/")

    trainB = load_images(mainPath + "trainB/")
    testB = load_images(mainPath + "testB/")

    np.savez_compressed(mainPath + "h2z_train.npz", trainA, trainB)
    np.savez_compressed(mainPath + "h2z_test.npz", testA, testB)
    
def load_compressed_images():
    """
    Charge les images depuis la version compressée fabriquée plus haut
    """
    dataTrain = np.load('Horse2Zebra/h2z_train.npz')
    dataTest = np.load('Horse2Zebra/h2z_test.npz')
    print("File loaded")
    XATrain = conv_255_1(dataTrain['arr_0'])
    print("Train A {} loaded".format(str(dataTrain['arr_0'].shape)))
    XBTrain = conv_255_1(dataTrain['arr_1'])
    print("Train B {} loaded".format(str(dataTrain['arr_1'].shape)))
    XATest = conv_255_1(dataTest['arr_0'])
    print("Test A {} loaded".format(str(dataTest['arr_0'].shape)))
    XBTest = conv_255_1(dataTest['arr_1'])
    print("Test B {} loaded".format(str(dataTest['arr_1'].shape)))
    return XATrain, XBTrain, XATest, XBTest

def conv_255_1(X):
    return X/127.5-1
def conv_1_255(X):
    return (X+1)*127.5

def show_data(X):
    img = conv_1_255(X)
    plt.clf()
    for i in range(img.shape[0]):
        plt.subplot(1, img.shape[0],i+1)
        plt.axis('off')
        plt.imshow(img[i].astype('uint8'))
    plt.show()


# ### Création des réseaux 

# In[ ]:


def new_discriminator(dim = 256, depht = 32, name=""):
    """
    Retourne un nouveau discriminator non compilé
    """
    input_layer = keras.layers.Input(shape=(dim, dim,3))
    
    #Layer 1 : Convolution avec un filtre de 4x4 qui se déplace de 2 pixels en 2 -> Division du nombre de pixel par 2; depht filtres utilisés
    #On ajoute un BatchNormalization pour réduire les poids et éviter une explosion du gradient
    #1] Conv; dim*dim*3 -> dim/2*dim/2*depht
    D = keras.layers.Conv2D(depht, (4,4), strides=(2,2), padding="same", input_shape=(dim,dim,3))(input_layer)
    D = keras.layers.BatchNormalization(momentum=0.8)(D)
    D = keras.layers.LeakyReLU(alpha = 0.2)(D) 

    #2] Conv; dim/2*dim/2*depht -> dim/4*dim/4*2*depht
    D = keras.layers.Conv2D(2*depht, (4,4), strides=(2,2), padding="same")(D)
    D = keras.layers.BatchNormalization(momentum=0.8)(D)
    D = keras.layers.LeakyReLU(alpha = 0.2)(D) 

    #3] Conv; dim/4*dim/4*2*depht -> dim/8*dim/8*4*depht
    D = keras.layers.Conv2D(4*depht, (4,4), strides=(2,2), padding="same")(D)
    D = keras.layers.BatchNormalization(momentum=0.8)(D)
    D = keras.layers.LeakyReLU(alpha = 0.2)(D) 

    #4] Conv; dim/8*dim/8*4*depht -> dim/16*dim/16*8*depht
    D = keras.layers.Conv2D(8*depht, (4,4), strides=(2,2), padding="same")(D)
    D = keras.layers.BatchNormalization(momentum=0.8)(D)
    D = keras.layers.LeakyReLU(alpha = 0.2)(D) 

    #5] Conv; dim/16*dim/16*8*depht -> dim/16*dim/16*8*depht
    D = keras.layers.Conv2D(8*depht, (4,4), strides=(1,1), padding="same")(D)
    D = keras.layers.BatchNormalization(momentum=0.8)(D)
    D = keras.layers.LeakyReLU(alpha = 0.2)(D) 

    #6] Dense; dim/16*dim/16*8*depht -> 1
    D = keras.layers.Flatten()(D)
    D = keras.layers.Dense(1)(D)
    D = keras.layers.Activation("sigmoid")(D) 
    
    return keras.Model(input_layer, D, name="d_{}".format(name))


# In[ ]:


def new_generator(dim = 256,depht = 32, n_resnet = 9, name=""):
    """Créer un générateur typique utilisé dans la suite
    La structure est plus complexe, on part de la taille d'une image que l'on compress, puis on réétend pour refaire une image.
    """
    input_layer = keras.layers.Input(shape=(dim,dim,3))

    #1] Conv; dim*dim*3 -> dim/2*dim/2*depht
    g = keras.layers.Conv2D(depht, (7,7), strides=(2,2), padding="same")(input_layer)
    g = keras.layers.BatchNormalization(momentum=0.8)(g)
    g = keras.layers.LeakyReLU(alpha = 0.2)(g)

    #2] Conv; dim/2*dim/2*depht -> dim/4*dim/4*2*depht
    g = keras.layers.Conv2D(2*depht, (3,3), strides=(2,2), padding="same")(g)
    g = keras.layers.BatchNormalization(momentum=0.8)(g)
    g = keras.layers.LeakyReLU(alpha = 0.2)(g)

    #3] Conv; dim/4*dim/4*2*depht -> dim/8*dim/8*4*depht
    g = keras.layers.Conv2D(4*depht, (3,3), strides=(2,2), padding="same")(g)
    g = keras.layers.BatchNormalization(momentum=0.8)(g)
    g = keras.layers.LeakyReLU(alpha = 0.2)(g)

    #Au milieu, on ajoute autant de resnet_block que l'on vezut
    for _ in range(n_resnet):
        g = create_resnet(n_filters=4*depht, T = g)
    
    #On redéroule dans l'autre sens
    #4] ConvT; dim/8*dim/8*4*depht -> dim/4*dim/4*2*depht
    g = keras.layers.Conv2DTranspose(2*depht, (3,3), strides=(2,2), padding="same")(g)
    g = keras.layers.BatchNormalization(momentum=0.8)(g)
    g = keras.layers.LeakyReLU(alpha = 0.2)(g)

    #5] ConvT; dim/4*dim/4*2*depht -> dim/2*dim/2*depht
    g = keras.layers.Conv2DTranspose(depht, (3,3), strides=(2,2), padding="same", activation="relu")(g)
    g = keras.layers.BatchNormalization(momentum=0.8)(g)
    g = keras.layers.LeakyReLU(alpha = 0.2)(g)

    #6] ConvT; dim/2*dim/2*depht -> dim*dim*3
    g = keras.layers.Conv2DTranspose(3, (7,7), strides=(2,2), padding="same")(g)
    g = keras.layers.Activation("tanh")(g)
    
    return keras.Model(input_layer, g, name="gen_{}".format(name))


def create_resnet(n_filters, T):
    N = keras.layers.Conv2D(n_filters, (3,3), strides=(1,1), padding="same")(T)
    N = keras.layers.BatchNormalization(momentum=0.8)(N)
    N = keras.layers.LeakyReLU(alpha = 0.2)(N)

    N = keras.layers.Conv2D(n_filters, (3,3), strides=(1,1), padding="same")(N)
    N = keras.layers.BatchNormalization(momentum=0.8)(N)

    #On additionne l'entrée et la sortie (ce qui fait la particularité du RESNET)
    N = keras.layers.Add()([N, T])

    #Dernière fonction d'activation
    N = keras.layers.LeakyReLU(alpha = 0.2)(N)
    return N


# In[ ]:


def create_training_gen(gen2vers1, d1, gen1vers2, dim = 256, name=""):
    """
    Cette méthode combine les différents réseau pour en déduire des fonctions de loss que nous allons chercher a minimiser
    Ici, c'est seulement gen2vers1 qui va être minimisé
    """
    gen2vers1.trainable = True
    gen1vers2.trainable = False
    d1.trainable = False
    
    #Entrées de notre réseau
    input_from_1 = keras.layers.Input(shape=(dim,dim,3))
    input_from_2 = keras.layers.Input(shape=(dim,dim,3))

    #Entrainement 1 : On veut que gen_a soit capable de tromper le discriminateur
    #On entraine donc le reseau d1(gen2vers1) en lui donnant des images du monde 2 et on cherche à ce que
    #cela atteigne 1, car on veut tromper d1 (qui n'est pas modifiable on le rappel)
    out_d1 = d1(gen2vers1(input_from_2))

    #Entrainement 2 et 3 : Le but est que gen2vers1 = gen1v2^-1
    #Il faut donc entrainer de facon a ce que gen2vers1(gen1vers2) = id et gen1vers2(gen2vers1) = id
    #On appelle cette boucle un cycle
    out_cycle1 = gen2vers1(gen1vers2(input_from_1))
    out_cycle2 = gen1vers2(gen2vers1(input_from_2))

    #Entrainement 4 : Enfin, il est important que gen2vers1 soit identité pour un élément venant deja du monde d'arrivée
    out_id1 = gen2vers1(input_from_1)
    
    #On fabrique alors le modèle
    return keras.Model([input_from_1, input_from_2], [out_d1, out_cycle1, out_cycle2, out_id1],name="trainGen{}".format(name))


# ### Méthodes utiles

# In[ ]:


def get_random_element(X, n):
    return X[np.random.randint(0,X.shape[0], n),...]


# ### Déroulement du programme

# In[ ]:


XATrain, XBTrain, XATest, XBTest = load_compressed_images()


# In[ ]:


#On créer nos deux discriminateurs
dA, dB = new_discriminator(dim = 256, depht = 32, name="A"),new_discriminator(dim = 256, depht = 32, name="B")

#On compile les deux discriminateurs avant de les désactiver pour la suite
opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)

dA.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
dB.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

#On créer nos deux générateurs
genBversA, genAversB = new_generator(dim = 256,depht = 32, n_resnet = 9, name="BversA"),new_generator(dim = 256,depht = 32, n_resnet = 9, name="AversB")


# In[ ]:





# In[ ]:





# In[ ]:


#On fabrique les modèles combinés d'entrainement
trainGenBversA = create_training_gen(genBversA, dA, genAversB)
trainGenBversA.compile(loss=["binary_crossentropy", "mae", "mae", "mae"], optimizer=opt, metrics=["accuracy"])

trainGenAversB = create_training_gen(genAversB, dB, genBversA)
trainGenAversB.compile(loss=["binary_crossentropy", "mae", "mae", "mae"], optimizer=opt, metrics=["accuracy"])


# In[ ]:





# In[ ]:


batch = 3
epochs = 3
N = max(XATrain.shape[0], XBTrain.shape[0])
run_by_epochs = int(N/batch)
run_by_epochs = 10

for e in range(epochs):
    print("next")
    for _ in tqdm(range(run_by_epochs)):
        print(_)
        #On commence l'entrainement
        xa_real, ya_real = get_random_element(XATrain, batch), np.ones(batch)[...,np.newaxis]
        xb_real, yb_real = get_random_element(XBTrain, batch), np.ones(batch)[...,np.newaxis]

        xa_fake, ya_fake = genBversA.predict(xb_real), np.zeros(batch)[...,np.newaxis]
        xb_fake, yb_fake = genAversB.predict(xa_real), np.zeros(batch)[...,np.newaxis]

        # On entraine le premier générateur genBversA
        # trainGenBversA = create_training_gen(genBversA, dA, genAversB)
        # create_training_gen(gen2vers1, d1, gen1vers2, dim = 256, name="")
        # [input_from_1, input_from_2], [out_d1, out_cycle1, out_cycle2, out_id1]
        trainGenBversA.train_on_batch([xa_real, xb_real], [yb_real, xa_real, xb_real, xa_real])

        # On entraine le deuxieme générateur genAversB
        # trainGenAversB = create_training_gen(genAversB, dB, genBversA)
        # create_training_gen(gen2vers1, d1, gen1vers2, dim = 256, name="")
        # [input_from_1, input_from_2], [out_d1, out_cycle1, out_cycle2, out_id1]
        trainGenAversB.train_on_batch([xb_real, xa_real], [ya_real, xb_real, xa_real, xb_real])

        #Et on entraine à présent les deux discriminateurs
        xa, ya = np.concatenate((xa_real, xa_fake)), np.concatenate((ya_real, ya_fake))
        dA.train_on_batch(xa, ya)

        xb, yb = np.concatenate((xb_real, xb_fake)), np.concatenate((yb_real, yb_fake))
        dB.train_on_batch(xb, yb)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




