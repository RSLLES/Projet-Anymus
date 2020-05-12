#Les données sont sur le site https://myanimelist.net/character.php
import requests
from bs4 import BeautifulSoup
import urllib.request


#Caractéristiques
i_page_debut = 0
n_pages_a_scrap = 10000
urls = ["https://myanimelist.net/character.php?limit={}".format(50*i) for i in range(i_page_debut, n_pages_a_scrap)]

for url in urls:
    page = requests.get(url)
    soup = BeautifulSoup(page.content, features="html.parser")
    all_characters = soup.find_all("tr", {"class": "ranking-list"})

    for chara in all_characters:
        bio = chara.find("td", {"class" : "people"})
        small_image_link, name = bio.find("img")['data-src'], bio.find("img")['alt']
        print(name)
        code, name = small_image_link.split('/')[7],small_image_link.split('/')[8].split('?')[0]
        big_image_link = "https://cdn.myanimelist.net/images/characters/{}/{}".format(code, name)
        urllib.request.urlretrieve(big_image_link, "Manga/Data/{}-{}.jpg".format(code, name))
