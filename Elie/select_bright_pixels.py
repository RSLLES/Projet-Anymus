# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 19:48:43 2020

@author: eliel
"""

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