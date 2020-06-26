# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 19:48:43 2020

@author: eliel
"""

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