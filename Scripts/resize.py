import os
from PIL import Image
import glob

i = 0
size = (1024,1024)

for f in glob.glob('*.jpg'):
    im = Image.open(f)
    w,h = im.size

    top, bottom = h//2 -1, h//2 + 1
    left, right = w//2 - 1, w//2 + 1

    #On augmente le carré tant que l'on peut
    while top > 0 and bottom < h and left > 0 and right < w:
        top -=1
        bottom += 1
        left -= 1
        right += 1

    #On crope
    im = im.crop((left, top, right, bottom)) 

    #On save
    im.thumbnail(size, Image.ANTIALIAS)
    im.save(f, quality=95)
    i+=1
    print("[{}]Done {}".format(i,f))