#! /bin/sh
ls
echo $PWD
dest="/d/HDD_Documents/Mines/Cours/1A/Projet_anime/TrainFaces/"
cd "105_classes_pins_dataset/"
find . -type f | \
    shuf -n 25000 | \
    while [ $(du -ks "$dest" | awk '{ print $1 }') -lt 10485760 ] && IFS= read -r fn; do
        cp "$fn" "$dest"
    done
read -p "Appuyer sur une touche pour continuer ..."