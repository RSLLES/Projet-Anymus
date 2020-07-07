# Pour constituer un dataset plus lumineux, cet algorithme effectue un trie parmi les images de myazaki : 

import shutil
from glob import glob 
from random import randint
import cv2
import numpy as np
from skimage import io

W = glob("C:/Users/eliel/OneDrive/Bureau/GAN/datasets/landscape2myazaki/trainA/*")
count = 0
for k in W:
    count+=1
    img = io.imread(k)[:, :, :-1]
    average = img.mean(axis=0).mean(axis=0)
    if average[0] + average[1] > 200:
        try:
            shutil.copyfile(k, f"C:/Users/eliel/OneDrive/Bureau/GAN/datasets/landscape2myazaki_filtered/trainA/test{count}.jpg")
        except:
            print("zob")