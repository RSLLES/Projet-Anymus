# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 18:38:31 2020

@author: eliel
"""

import numpy as np
import cv2
import cv2 as cv
from tqdm import tqdm

nb_im = 5248 + 1
tmoy = 0

for iter in tqdm(range(1784, nb_im)):
    try:
        img_ = cv.imread(f"C:/Users/eliel/OneDrive/Bureau/Myazaki_data/image{iter}.jpg")
        img = cv2.resize(img_,(512,512))
        edges = cv.Canny(img,150,150)
        kernel = np.ones((3,3))
        dilation = cv2.dilate(edges,kernel,iterations = 1)
        blur = cv.GaussianBlur(img,(7,7),cv2.BORDER_DEFAULT)
        result = np.zeros((512,512,3))
        for i in range(512):
            for j in range(512):
                for u in range(3):
                    k = u
                    test = (dilation[i][j] == 255) * 1 #pas supprimer le *1
                    result[i][j][u] = blur[i][j][k]*test + img[i][j][k]*(1-test)
        result = result.astype(int)
        cv2.imwrite(f"smooth{iter}.jpg", result)
    except:
        print(f"{iter} failed")