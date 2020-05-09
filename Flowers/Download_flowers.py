import urllib.request
from PIL import Image

standard_size = 128

def download_image(u, path):
    try:
        urllib.request.urlretrieve(u, path)
        return True
    except:
        print("ERREUR AVEC " + u)
    return False

def standard(path):
    img = Image.open(path)
    width, height = img.size  # Get dimensions

    if width > standard_size and height > standard_size:
        # keep ratio but shrink down
        img.thumbnail((width, height))

    # check which one is smaller
    if height < width:
        # make square by cutting off equal amounts left and right
        left = (width - height) / 2
        right = (width + height) / 2
        top = 0
        bottom = height
        img = img.crop((left, top, right, bottom))

    elif width < height:
        # make square by cutting off bottom
        left = 0
        right = width
        top = 0
        bottom = width
        img = img.crop((left, top, right, bottom))

    if width > standard_size and height > standard_size:
        img.thumbnail((standard_size, standard_size))

    #Dernier check
    if img.size != (128,128):
        raise ValueError

    img.save(path)
    

    


with open("Flowers/flowers_url.txt") as f:
    id = 0
    for url in f.readlines():
        path = "Flowers/Data/" + str(id) + ".jpg"

        #On télécharge l'image
        if(download_image(url, path)):
            #On la redimensionne
            try:
                standard(path)
                #On passe a la suivante
                id += 1
            except:
                print("Erreur de traitement avec")