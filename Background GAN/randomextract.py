# Petit code qui permet de facilement extraire des images d'un dataset pour former un training set

import shutil
from glob import glob 
from random import randint

W = glob("C:/Users/eliel/OneDrive/Bureau/GAN/datasets/landscape2myazaki_filtered/trainA/*")

for k in range(150):
    i = randint(0,len(W)-1)
    try:
        shutil.move(W[i], f"C:/Users/eliel/OneDrive/Bureau/GAN/datasets/landscape2myazaki_filtered/testA/test{k}.jpg")
    except:
        print("zob")