from PIL import Image
import numpy as np
from itertools import product
from tqdm import tqdm

d_min = 10 # Doit etre < c, sinon deja ca n'a aucun sens et après y'a une belle boucle infinie
c = 128

def search_for_d(L,h,c):
    d = d_min
    # La terminaison est évidente comme d = c-1 est sol triviale 
    # puisqu'alors c-d=1 et 1 divise tout nombre entier. 
    # On peut donc mettre une boucle infinie pour le swag.
    while True:
        if (L-d)%(c-d) == 0 and (h-d)%(c-d) == 0:
            return d, (L-d)//(c-d), (h-d)//(c-d)
        d+=1

def decoupage():
    im = Image.open("full_hd.jpg")
    L,h = im.size
    pix = np.array(im)

    d, nL, nh = search_for_d(L,h,c)

    All_images = []
    for a,b in tqdm(product(range(nL), range(nh))):
        jm, jM = b*(c-d), (b+1)*(c-d)+d
        im, iM = a*(c-d), (a+1)*(c-d)+d
        crop_img = Image.fromarray(pix[jm:jM, im:iM, ...].astype(np.uint8))
        crop_img.save("dissambled/{}.{}.jpg".format(b,a))
decoupage()

def fusion(L,h):
    d, nL, nh = search_for_d(L,h,c)
    pass
