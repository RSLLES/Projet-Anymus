import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import tensorflow.keras as keras
import time


print(tf.__version__)
#Caractéristiques

class GAN:
    def __init__(self):
        self.dim = 128 #Dimension de notre image, qui doit être carrée
        self.depth = 32 #nombre de base de filtre dans nos couche de convolution
        self.dropout = 0.4 #pourcentage de neurone désactivé par entrainement pour éviter de l'overfitting
        self.n = 645 #Nombre d'images de fleurs que l'on a
        self.nFake = 645 #Nombre de fausses images que l'on va créer pour le test
        self.trainSteps = 500

    def create_discriminator(self):
        self.D = keras.models.Sequential()

        #Couche 1 : couche de convolution, divise par 2 la taille (on se retrouve avec du 64*64), on prend depth filtres de taille 3*3
        #Il faut spécifier la forme de l'entrée comme c'est la première couche du réseau
        #On ajoute un dropout pour éviter de surentrainer
        self.D.add(keras.layers.Conv2D(self.depth, (3,3), strides = (2,2) , padding='same', activation='relu', input_shape=(self.dim, self.dim, 3)))
        self.D.add(keras.layers.Dropout(self.dropout))

        #Couche 2 : convolution, /2, 2*depths on est donc en 32*32*64
        self.D.add(keras.layers.Conv2D(2*self.depth, (3,3), strides=(2,2), padding="same", activation="relu"))
        self.D.add(keras.layers.Dropout(self.dropout))

        #Couche 3 : convolution, /2, 4*depths : oin arrive à 16*16*128
        self.D.add(keras.layers.Conv2D(4*self.depth, (3,3), strides=(2,2), padding="same", activation="relu"))
        self.D.add(keras.layers.Dropout(self.dropout))

        #Couche 4 : convolution, /2, 8*depths : oin arrive à 8*8*256
        self.D.add(keras.layers.Conv2D(8*self.depth, (3,3), strides=(2,2), padding="same", activation="relu"))
        self.D.add(keras.layers.Dropout(self.dropout))

        #Dernière couche, dense layer en discrimination, avec une sigmoid
        #On reduit tout en un vecteur avant
        self.D.add(keras.layers.Flatten())
        self.D.add(keras.layers.Dense(1, activation="sigmoid"))

        #Synthèse
        self.D.summary()

    def create_generator(self):
        self.G = keras.models.Sequential()

        #Couche 1 : 100 -> dim/8 * dim/8 * 4*depth
        self.G.add(keras.layers.Dense(int(self.dim/8)*int(self.dim/8)*4*self.depth, input_dim=100))
        self.G.add(keras.layers.BatchNormalization(momentum=0.9))
        self.G.add(keras.layers.Activation("relu"))
        self.G.add(keras.layers.Reshape((int(self.dim/8),int(self.dim/8),4*self.depth)))
        self.G.add(keras.layers.Dropout(self.dropout))

        #Couche 2 :  dim/8 * dim/8 * 4*depth -> dim/4 * dim/4 * 2*depth
        #UpSampling permet d'augmenter la taille du vecteur en interpollant les valeurs manquantes
        self.G.add(keras.layers.UpSampling2D((2,2)))
        self.G.add(keras.layers.Conv2DTranspose(2*self.depth, (3,3), padding="same"))
        self.G.add(keras.layers.BatchNormalization(momentum=0.9))
        self.G.add(keras.layers.Activation("relu"))

        #Couche 3 : dim/4 * dim/4 * 2*depth -> dim/2 * dim/2 * depth 
        #UpSampling permet d'augmenter la taille du vecteur en interpollant les valeurs manquantes
        self.G.add(keras.layers.UpSampling2D((2,2)))
        self.G.add(keras.layers.Conv2DTranspose(self.depth, (3,3), padding="same"))
        self.G.add(keras.layers.BatchNormalization(momentum=0.9))
        self.G.add(keras.layers.Activation("relu"))

        #Couche 3 : dim/2 * dim/2 * depth -> dim * dim * depth
        #UpSampling permet d'augmenter la taille du vecteur en interpollant les valeurs manquantes
        self.G.add(keras.layers.UpSampling2D((2,2)))
        self.G.add(keras.layers.Conv2DTranspose(self.depth, (3,3), padding="same"))
        self.G.add(keras.layers.BatchNormalization(momentum=0.9))
        self.G.add(keras.layers.Activation("relu"))

        #Couche 4 : dim * dim * depth -> dim*dim*3
        self.G.add(keras.layers.Conv2DTranspose(3, 5, padding='same'))
        self.G.add(keras.layers.Activation('sigmoid')) #Pour avoir une image entre 0 et 1

        #Résumé
        self.G.summary()

    def create_training_model_discriminator(self):
        #On fabrique le model pour entrainer le discriminateur. Rien de bien dur, 
        self.TD = keras.models.Sequential()
        #Optimizer, valeurs complétement arbitraires
        optimizer = keras.optimizers.RMSprop(lr=0.008, clipvalue=1.0, decay=6e-8)
        self.TD.add(self.D)

        self.TD.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    def create_training_model_GAN(self):
        #La, il faut entrainer les deux réseaux en même temps.
        #On construit donc la structure Generateur -> Disciminateur que l'on va optimiser
        optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)
        self.TGAN = keras.Model.Sequential()
        self.TGAN.add(self.G())
        self.TGAN.add(self.D())
        self.TGAN.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def load_imgs(self):
        self.imgs = np.empty((self.n,self.dim, self.dim, 3))
        for k in range(self.imgs.shape[0]):
            self.imgs[k,...] = mpimg.imread("Flowers/Data/"+str(k)+".jpg")/255


    def show_img(self, img):
        imgplot = plt.imshow(img)
        plt.show()

    def create_training_sets(self):
        #On fabrique les faux
        self.fakes = np.random.uniform(0,1,size=(self.nFake,self.dim, self.dim, 3))
        
        #On créer notre datatest
        self.x = np.concatenate((self.imgs, self.fakes))
    
        #On créer les réponses à la question "est ce que cette image est vraie? pour entrainer le discrimineur
        self.y = np.concatenate((np.array([1 for k in range(self.n)]), np.array([0 for k in range(self.nFake)])))
        
    def train_discriminator(self):
        for _ in range(self.trainSteps):
            s = time.time()
            d_loss = self.TD.train_on_batch(self.x, self.y)
            print(str(d_loss) + " | " + str(time.time()-s) +  "s")
    



#On créer notre GAN
gan = GAN()
gan.create_discriminator()
gan.create_training_model_discriminator()
gan.load_imgs()
gan.create_training_sets()
print(gan.train_discriminator())


