# Projet Animus

Par ESCRIBE Florent, LANNELONGUE Vincent, LESEC Élie, SÉAILLES Romain

## Synthèse du sujet proposé
L'intitulé du sujet portait sur le transfert de style.
Notre encadrant nous a proposé un sujet assez large : le but était d'imaginer un transfert d’image
ou de dessin (crayonné, bande dessinée...) vers un rendu manga. Il nous a conseillé de
restreindre le sujet, ne pas être trop ambitieux pour que nous ayons un résultat à la fin. Il nous a
donné plusieurs pistes de travaux, et a été très ouvert à nos propositions. Dès le début, le sujet
était orienté machine learning et réseaux de neurones, ce qui a par la suite conditionné notre
approche.

## Notre interprétation du sujet, notre objectif

La première idée était de transformer un crayonné en un dessin de style manga. On s’est très vite
concentré sur le visage pour faciliter un peu l'approche, entre autre pour avoir une base de donnée
assez cohérente (photos/dessins cadrées de la même façon) , et pour ne pas vouloir trop en faire.
Notre volonté de s’attaquer spécifiquement au visage vient aussi du fait que la forme du visage est
ce qui fait la grande particularité du style Manga par des caractéristiques fortes (gros yeux, tête
plus large, nez inexistant, etc …). Ce choix nous semble pertinent car en pratique il pourrait servir
d’assistant aux mangakas pour imaginer des visages, ou bien par exemple cela pourrait
déboucher sur une application “fun” pour des utilisateurs lambdas.
Par la suite, nous nous sommes concentrés sur la transformation d’un visage humain depuis une
photo classique en un dessin '' manga-isé '', le dessin crayonné étant problématique pour plusieurs
raisons détaillées plus tard.

## Choix techniques et algorithmiques

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

## Niveau de réalisation actuel

Les concepts algorithmiques sont bien appréhendés, mais les résultats ne sont pas encore au
rendez-vous.
Nous avons pour l'instant réussi à former des bases de données satisfaisantes pour entraîner nos
réseaux, et nous avons réussi la partie manipulation d’images par python, cruciale pour intégrer
nos dataset. Nous nous sommes alors entraînés à maîtriser la librairie Tensorflow en réalisant
plusieurs réseaux de neurones de plus en plus sophistiqués. Nous avons réussi à reproduire des
structures plus complexes comme des CGAN et à les modifier selon les conseils de plusieurs
articles afin de mieux prendre en compte la spécificité du problème (faciliter la déformation des
yeux plus que le simple changement de texture entre autre).
En parallèle du travail sur les CGAN, nous avons entraîné un discriminateur hors-GAN afin de
vérifier la cohérence de notre approche. Il s’agit d’un CNN classique entraîné sur une grande base
de données de visages et de mangas (32 000 photos au total), utilisant des techniques de dropout,
de régularisation, pour distinguer si l'image en entrée est une photo ou un dessin de manga.
L'objectif de ce travail est de savoir si la base de données que nous avons arrive à bien
caractériser photos de visages et visages mangas en général. Le but étant bien sûr d'intégrer cette
architecture dans un GAN très vite.

## Problèmes rencontrés et remarques

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
