#!git clone https://github.com/timesler/facenet-pytorch facenet_pytorch

from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face  # modèles préentrainés
import torch
from torch.utils.data import DataLoader 
from torchvision import datasets
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image, ImageDraw # gestion des images

def seg_faces(path):
    # On définit les modèles
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    mtcnn = MTCNN ()
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # on récupère l'image et on récupère les boîtes
    path = 'img_test.jpg'
    img = Image.open(path)
    boxes, probs, points = mtcnn.detect(img, landmarks = True) 
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    n=0
    for i in range (len(boxes)):
        if probs[i] > 0.95:
            draw.rectangle(boxes[i], width=5)
            n+=1

    #print('n =', n)
    #final = np.array(img_draw)
    #plt.imshow(final)

    # On agrandit les boîtes pour englober toute la tête
    img = np.array(img)
    coord = []
    boxes_up = boxes.copy()
    p = 0.3    # pourcentage d'augmentation 
    for box in boxes_up:
        l = box[2] - box[0]
        h = box[3] - box[1]
        coord_x = box[0] + l/2
        coord_y = box[1] + h/2
        coord.append([coord_x, coord_y])
        box[0] -= l*p 
        box[2] += l*p
        box[1] -= h*p
        box[3] += h*p

    coord = np.array([coord])
    coord = coord.astype(int)
    boxes_up = boxes_up.astype(int)


    # On crée un array des visages découpés
    faces = []
    for box in boxes_up:
        face = np.array(img[max(0,box[1]):box[3], max(0, box[0]):box[2], ...])
        faces.append(face)

    faces = np.array(faces)
    
    return faces, coord