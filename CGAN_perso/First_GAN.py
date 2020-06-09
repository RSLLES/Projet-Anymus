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
        self.n = 1300 #Nombre d'images de fleurs que l'on a
        self.nFake = 976 #Nombre de fausses images que l'on a
        self.trainSteps = 300
        self.half_batch_size = 40 #Moitié du batch size, ce qui correspond au nombre de vrais photos prisent pour le test
        self.batch_size = 2*self.half_batch_size #Nombre d'images utilisés pour une iteration

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
        #On fabrique le model pour entrainer le discriminateur. Rien de bien dur, puisque le le discriminateur s'entraine tout seul
        self.TD = keras.models.Sequential()
        #Optimizer, valeurs complétement arbitraires
        optimizer = keras.optimizers.RMSprop(lr=0.008, clipvalue=1.0, decay=6e-8)
        self.TD.add(self.D)

        self.TD.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    def create_training_model_GAN(self):
        #La, il faut entrainer les deux réseaux en même temps.
        #On construit donc la structure Generateur -> Disciminateur que l'on va optimiser
        optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)
        self.TGAN = keras.models.Sequential()
        self.TGAN.add(self.G)
        self.TGAN.add(self.D)
        self.TGAN.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def load_img(self, path):
        return mpimg.imread(path)/255

    def load_imgs(self):
        self.imgs = np.empty((self.n, self.dim, self.dim, 3))
        for k in range(self.n):
            self.imgs[k,...] = self.load_img("Flowers/Data/"+str(k)+".jpg")
        print("{:d} images chargées".format(self.n))

    def load_fakes(self):
        self.fakes = np.empty((self.nFake, self.dim, self.dim, 3))
        for k in range(self.nFake):
            self.imgs[k,...] = self.load_img("Flowers/Random/"+str(k)+".jpg")
        print("{:d} images chargées".format(self.nFake))


    def show_img(self, img, title=""):
        imgplot = plt.imshow(img)
        plt.title = title
        plt.legend()
        plt.show()

    def create_training_sets(self):
        #On fabrique half_batch_size faux
        self.noise = np.random.uniform(0,1,size=(self.half_batch_size,self.dim, self.dim, 3))

        #On tire half_batch_size vrais images au hasard dans la liste que l'on a
        self.vrais = self.imgs[np.random.randint(0,self.n, size=self.half_batch_size) ,...]

        #On tire half_batch_size fausses images au hasard dans la liste que l'on a
        self.f = self.fakes[np.random.randint(0,self.nFake, size=self.half_batch_size) ,...]
        
        #On créer notre datatest
        self.x = np.concatenate((self.vrais, self.f, self.noise))
    
        #On créer les réponses à la question "est ce que cette image est vraie? pour entrainer le discrimineur
        self.y = np.concatenate((np.array([1 for k in range(self.half_batch_size)]), np.array([0 for k in range(2*self.half_batch_size)])))
        
    def train_discriminator(self):
        s = time.time()
        for _ in range(self.trainSteps):
            #On fabrique nos training sets
            self.create_training_sets()

            #On lance la procédure
            d_loss = self.TD.train_on_batch(self.x, self.y)

            #On affiche le résultat
            if _%(self.trainSteps//10) == 0:
                print("Loss : {:d}, Précision : {:d}, Temps : {:d}".format(d_loss[0], d_loss[1], time.time()-s))
                s = time.time()

    def train_all(self):
        s = time.time()
        for _ in range(self.trainSteps):
            #On fabrique nos training sets
            self.create_training_sets()

            #On lance l'entrainement du discriminateur
            d_loss = self.TD.train_on_batch(self.x, self.y)

            #On lance l'entrainement du générateur
            x = np.random.uniform(0,1,size=(self.batch_size,100))
            y = np.array([1 for k in range(self.batch_size)])
            g_loss = self.TGAN.train_on_batch(x, y)

            #On affiche le résultat
            if _%(self.trainSteps//10) == 0:
                print("Discriminateur | Loss : {}, Précision : {}, Temps : {}".format(d_loss[0], d_loss[1], time.time()-s))
                print("GAN | Loss : {}, Précision : {}, Temps : {}".format(g_loss[0], g_loss[1], time.time()-s))
                print("-----------------------------------")
                s = time.time()

    def save_weigts(self):
        self.TGAN.save_weights("TGAN.h5")

    def load_weigts(self):
        self.TGAN.load_weights("TGAN.h5")

    def save_weigts_discriminator(self):
        self.D.save_weights('D1.h5')

    def load_weigts_discriminator(self):
        self.D.load_weights('D1.h5')

    def test_discriminator_with_img(self, path):
        img = self.load_img(path)
        img_transform = img[np.newaxis, ...]
        result  = self.TD.predict(img_transform)
        print(result)
        self.show_img(img, str(result))

    def generate_img(self):
        x = np.random.uniform(0,1,size=(1,100))
        result = self.G.predict(x)
        self.show_img(result[0,...])
    


gan = GAN()
gan.create_discriminator()
gan.create_generator()
gan.create_training_model_discriminator()
gan.create_training_model_GAN()
gan.load_imgs()
gan.load_fakes()
gan.train_all()
gan.save_weigts()
#gan.load_weigts()
gan.generate_img()
