import os
import numpy as np
import matplotlib.pyplot as plt

def Load_G_AB(path, g_AB):
    if (os.path.isfile(path)):
        g_AB.load_weights(path)
        print("Weights loaded")
        return True
    else:
        print("Impossible de charger les poids du réseau. Il doit se trouver dans '{}'".format(path))
        return False

def Load_Weights(wf, d_A, d_B, g_AB, g_BA, aux_d_A, aux_d_B, aux_g_AB, aux_g_BA):
    """Sauvegarde les poids deja calculés, pour pouvoir reprendre les calculs plus tard si jamais"""
    os.makedirs(wf, exist_ok=True)
    if (os.path.isfile(wf + "d_A.h5") and os.path.isfile(wf + "g_AB.h5") 
    and os.path.isfile(wf + "aux_d_A.h5") and os.path.isfile(wf + "aux_g_AB.h5") 
    and os.path.isfile(wf + "d_B.h5") and os.path.isfile(wf + "g_BA.h5")
    and os.path.isfile(wf + "aux_d_B.h5") and os.path.isfile(wf + "aux_g_BA.h5")):
        d_A.load_weights(wf + "d_A.h5")
        d_B.load_weights(wf + "d_B.h5")
        g_AB.load_weights(wf + "g_AB.h5")
        g_BA.load_weights(wf + "g_BA.h5")
        aux_d_A.load_weights(wf + "aux_d_A.h5")
        aux_d_B.load_weights(wf + "aux_d_B.h5")
        aux_g_AB.load_weights(wf + "aux_g_AB.h5")
        aux_g_BA.load_weights(wf + "aux_g_BA.h5")
        print("Weights loaded")
    else:
        print("Missing weights files detected. Starting from scratch")

def save(wf, d_A, d_B, g_AB, g_BA, aux_d_A, aux_d_B, aux_g_AB, aux_g_BA):
    """Sauvegarde les poids deja calculés, pour pouvoir reprendre les calculs plus tard si jamais"""
    os.makedirs(wf, exist_ok=True)
    d_A.save_weights(wf + "d_A.h5")
    d_B.save_weights(wf + "d_B.h5")
    g_AB.save_weights(wf + "g_AB.h5")
    g_BA.save_weights(wf + "g_BA.h5")
    aux_d_A.save_weights(wf + "aux_d_A.h5")
    aux_d_B.save_weights(wf + "aux_d_B.h5")
    aux_g_AB.save_weights(wf + "aux_g_AB.h5")
    aux_g_BA.save_weights(wf + "aux_g_BA.h5")


def sample_images(data_loader, epoch, batch_i, dataset_name, g_AB, g_BA, heatmap_g_AB, heatmap_g_BA, heatmap_d_A, heatmap_d_B):
    os.makedirs('images/%s' % dataset_name, exist_ok=True)

    r, c = 2, 5

    imgs_A = data_loader.load_data(domain="A", batch_size=1, is_testing=True)
    imgs_B = data_loader.load_data(domain="B", batch_size=1, is_testing=True)

    # Rescale heatmap btw -1 and 1
    def rescale_hm(img):
        M,m = np.max(img), np.min(img)
        return 2*(img-m)/(M-m)-1

    # Translate images to the other domain
    fake_B = g_AB.predict(imgs_A)
    fake_A = g_BA.predict(imgs_B)
    # Translate back to original domain
    reconstr_A = g_BA.predict(fake_B)
    reconstr_B = g_AB.predict(fake_A)
    # Calculate heatmap
    hm_g_A = rescale_hm(heatmap_g_AB.predict(imgs_A))
    hm_g_B = rescale_hm(heatmap_g_BA.predict(imgs_B))
    hm_d_A = rescale_hm(heatmap_d_A.predict(fake_A))
    hm_d_B = rescale_hm(heatmap_d_B.predict(fake_B))

    titles = ['Original', 'Gen', 'Translated', 'Discr', 'Reconstructed']
    fig, axs = plt.subplots(r, c, dpi=200)

    def show_row(r_i, img, hm_g, fake, hm_d, reconstr):
        def show(i,j, im):
            axs[i,j].imshow(0.5*im[0,...]+0.5)
            axs[i,j].set_title(titles[j])
            axs[i,j].axis('off')
        # Image normale
        show(r_i, 0, img)
        # Heatmap gen
        show(r_i, 1, hm_g)
        # Fake
        show(r_i, 2, fake)
        # Heatmap disc
        show(r_i, 3, hm_d)
        # Reconstructed
        show(r_i, 4, reconstr)

    show_row(0, imgs_A, hm_g_A, fake_B, hm_d_B, reconstr_A)
    show_row(1, imgs_B, hm_g_B, fake_A, hm_d_A, reconstr_B)

    fig.savefig("images/%s/%d_%d.png" % (dataset_name, epoch, batch_i), dpi=200)
    plt.close()