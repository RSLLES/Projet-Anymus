import requests
from json import loads
import urllib.request
import os
from PIL import Image

#Caractéristiques
i_page_debut = 2
n_pages_a_scrap = 10000
urls = ["https://unsplash.com/napi/landing_pages/images?page={}&per_page=20".format(i) for i in range(i_page_debut, n_pages_a_scrap)]
i = 0
os.makedirs('Unsplash/', exist_ok=True)

for url in urls:
    page = requests.get(url)
    photos = loads(page.content)['photos']

    for photo in photos:
        name = photo['id']
        download_link = photo['urls']['regular']
        try:
            urllib.request.urlretrieve(download_link, "Unsplash/{}.jpg".format(name))
            try:
                im = Image.open("Unsplash/{}.jpg".format(name))
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
                im.thumbnail((1024,1024), Image.ANTIALIAS)
                im.save("Unsplash/{}.jpg".format(name), quality=95)
                i+=1
                print("[{}]Done {}".format(i,name))

            except:
                print("Erreur de traitement de {}".format(name))
                os.remove("Unsplash/{}.jpg".format(name))
        except:
            print("ERREUR de téléchargement de {}".format(name))
