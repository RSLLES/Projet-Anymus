# Projet Anymus

Par ESCRIBE Florent, LANNELONGUE Vincent, LESEC Élie, SÉAILLES Romain

## Présentation et discussion sur le sujet retenu
### Synthèse du sujet proposé
L'intitulé du sujet portait sur le transfert de style.
Notre encadrant nous a proposé un sujet assez large : le but était d'imaginer un transfert d’image
ou de dessin (crayonné, bande dessinée...) vers un rendu manga. Dès le début, le sujet
était orienté machine learning et réseaux de neurones, ce qui a par la suite conditionné notre
approche.

### Notre interprétation du sujet, notre objectif

La première idée était de transformer un crayonné en un dessin de style manga. On s’est très vite
concentré sur le visage pour faciliter un peu l'approche, entre autre pour avoir une base de donnée
assez cohérente (photos/dessins cadrées de la même façon), et pour ne pas vouloir trop en faire.
Notre volonté de s’attaquer spécifiquement au visage vient aussi du fait que la forme du visage est
ce qui fait la grande particularité du style Manga par des caractéristiques fortes (gros yeux, tête
plus large, nez inexistant, etc …). Ce choix nous semble pertinent car en pratique il pourrait servir
d’assistant aux mangakas pour imaginer des visages, ou bien par exemple cela pourrait
déboucher sur une application “fun” pour des utilisateurs lambdas.
Par la suite, nous nous sommes concentrés sur la transformation d’un visage humain depuis une
photo classique en un dessin '' manga-isé '', le dessin crayonné étant problématique pour plusieurs
raisons détaillées plus tard.

### Choix techniques et algorithmiques

Dans un premier temps, nous avons travaillé sur du transfert de style : en utilisant un CNN préentraînés à la classification d'images (VGG 16 par exemple), il est possible de caractériser le style
et le contenu d'une image, et ainsi de faire varier le premier tout en conservant une partie du
2ème. Mais ces algorithmes sont adaptés à des transferts de texture (ils marchent très bien pour
appliquer le style d’un artiste à une photo), et ne sont donc pas adapté à notre problème.
Nous avons donc abandonné ce genre de technique pour se focaliser sur des GAN (Generative
Adversarial Networks, ou réseau de neurone antagoniste génératif). Un GAN repose sur 2 CNN :
un générateur qui crée une image devant faire partie de la base de données visée, et un
discriminateur qui doit être capable de déterminer si une image fait partie cette base de données
ou non. Le générateur s'entraîne en essayant de berner le discriminateur, qui lui s'entraîne dans
l'objectif inverse. Ces algorithmes sont bien plus capables de transformer une image, que notre
1ère approche.
Plus précisément, nous nous sommes attardés sur des CGAN, ou GAN cycliques. L’idée est de
combiner 2 GAN pour pouvoir créer un convertisseur entre 2 bases de données, dans les 2 sens.
Cela permet d’avoir une plus grande cohérence entre l’image de départ et le résultat. Ces
algorithmes sont plus adaptés à du changement de forme que les algorithmes de transfert de style
mentionnés précédemment. Mais il faut malgré tout largement complexifier leur structure pour en
avoir un à la hauteur du problème auquel nous nous attaquons. On utilise pour cela de la dilated
convolution afin de prendre en compte le contenu de l'image à une échelle plus grande qu'avec
des filtres de convolution plus classiques.
Dans le but de créer des bases de données adaptées, nous avons aussi réalisé plusieurs scripts
python d’extraction de databases pertinentes depuis des sites spécifiques.

### Problèmes rencontrés et remarques

Il a été difficile de mesurer l’ampleur de la tâche au début du projet. Nous nous sommes très vite
rendu compte de la complexité du travail sur le dessin : transformer l’intégralité des éléments d’une
planche de bande-dessinée (visage, corps, accessoires, vêtements, arrière-plan) en manga est
une tâche extrêmement ardue et ne peut pas être effectuée par un seul réseau. Nous avons donc
décidé de commencer d’organiser notre travail de sorte qu’à n’utiliser qu’un seul réseau de
neurones et ne traiter qu’une seule partie du personnage : le visage (pour d'autres raisons
évoquées plus tôt).
Nous avons aussi abandonné le dessin crayonné assez vite car les algorithmes déjà existants et
entraînés avaient beaucoup de mal avec, et qu’il fallait adopter une autre approche pour gérer le
dessin. Cela vient notamment du fait que, contrairement à une photo, un dessin crayonné est très
vide (beaucoup de blanc).
Pour ce qui est de notre travail sur les GAN et CGAN, nous faisons encore face à de nombreux
problèmes bien connus de convergence sur les algorithmes utilisés. Dans le cas du CGAN :
effondrement des modes, entrainement trop rapide des discriminateurs...

## Utilisation
### Structure du projet
La dernière version du projet utilise l'architecture U-GAT-IT légèrement modifiée et implémentée à l'aide de la bibliothèque Keras.
```
.
├── _CGAN/
│   ├── custom_layer.py
│   ├── data_loader.py
│   ├── reseaux.py
│   ├── run.py
│   ├── train.py
│   ├── utils.py
│   └──_Trombi_results/
│      ├── result-19escrive.jpg
│      └── ...
.
```
L'algorithme de transformation des visages se trouve dans le dossier CGAN.
Il est divisé comme suit :
- custom_layer.py contient nos propres layers construient à l'aide de la bibliothèque Keras. Ce fichier contient notamment le layer à double sortie Aux permettant de créer la sortie auxilliaire caractéristique de l'architecture permettant de l'entrainer ainsi que la nouvelle fonction d'activation AdaLin à la fois controlée par des poids entrainable et par une entrée spécifique.
- data_loader.py Fichier repris et légèrement modifié du git de , permettant de gérer de facon particulièrement efficace l'importation d'images pour l'entrainement et la création de batch
- reseaux.py Fichier contenant nos architecture reseau pour nos discriminateurs et nos générateurs ainsi que le modèle combiné permettant d'entrainer les seconds au travers des premiers selon les 4 équations spécifiques des Cycle GAN : Tromper le discriminateur, 2*Cycle consistency et 1 fois Identity
- run.py Fichier permettant d'utiliser le reseau, expliqué plus bas
- train.py Fichier permettant d'entrainer le reseau, expliqué plus bas
- utils.py Contient quelques fonctions utiles notamment pour l'importation et la sauvegarde des poids des réseaux.
- Trombi_results/ Dossier contenant nos résultats en faisant tourné notre algorithme sur les photos de profils des élèves p18 et p19 du portail des élèves.

### Utilisation du réseau : run.py
Le paramètre help du fichier explique son fonctionnement :
```
>>> python .\run.py -h
usage: run.py [-h] [-o O] [-g G] [-p P] [-m] [-v] image

Permet de faire tourner le réseau de transformation Visage -> Manga sur des images.

positional arguments:
  image          Chemin vers l'image à traiter. Cela peut aussi etre un chemin vers un dossier, auquel cas toutes les
                 images trouvées dedans seront traitées

optional arguments:
  -h, --help     show this help message and exit
  -o O           Dossier dans lequel seront stockés les résultats. Par défaut 'results/'
  -g G           Chemin vers le fichier contenant les poids du générateur à utiliser. Par défaut 'g.h5'
  -p P           Ajoute le prefixe prefixe-nom_de_l'image_original.jpg au résultat. 'result' par défaut.
  -m, --mosaic   Pour traiter des images hd qui ne doivent pas être redimensionnées en 256x256. Elles seront alors
                 composées d'une mosaic de carrés de tailles 256x256.
  -v, --verbose  Affiche les logs de tensorflow
>>>
```
Comme expliqué, il est possible de donner en entrée du script à la fois une image précise à transformée, ou bien de lui donner un dossier auquel cas le script rechercera tous les images à l'intérieur et les traitera toutes.

Par défaut, toutes les images seront recadrés en 256x256 avant d'être modifié. Il est malgré tout possible de traiter des images plus grande à l'aide de l'argument mosaic, qui va alors subdiviser une grandes images en une série de carré d 256x256 pour es traiter indépendament et reconstruire par collage l'image originale à l'arrivée. Il est à noter que cette fonction fait sortir l'algorithme de son cadre d'utilisation normal et donc ne donne pas des résultats convaincant : c'est purement une fonction de test qui a été laissée dans la version finale.

Un générateur pré entrainé peut êre trouvé ici en libre accès :

Exemple d'utilisation :
```
>>> python .\run.py -g .\g.h5 -o .\results_live\ .\inputs_live\
Using TensorFlow backend.
Traitement du dossier .\inputs_live\ en 256x256
Weights loaded
100%|███████████████████████████████████| 9/9 [00:11<00:00,  1.25s/it]
>>>
```
Ce qui donne comme résultat les images proposées en introduction du README.

Il est aussi possible de l'utiliser sur une séquence d'image puis de recompiler le tout pour en faire un .gif amusant :
```
>>> python .\run.py -o .\Gatsby_output\ -g .\g.h5 .\Gatsby\
Using TensorFlow backend.
Traitement du dossier .\Gatsby\ en 256x256
Weights loaded
100%|███████████████████████████████████| 89/89 [00:13<00:00,  6.47it/s]
>>>
```
Ce qui donne comme résultat après avoir recompilé les images avec ffmpeg en un gif:
![The great Gatsby](README_Illustration/gatsby.gif)