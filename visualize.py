"""
wontak ryu
ryu071511@gmail.com
visualize the mixup reuslt.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from hyper_params import get_hyper_params
from tensorflow.python.keras.datasets import mnist, cifar10, fashion_mnist
from module import mixup_data_one_sample, mixup_data
from sklearn.decomposition import PCA
import imageio
import matplotlib
matplotlib.use('TKAgg')
hyper_params = get_hyper_params()


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()

    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = np.reshape(buf, (h, w, 3))
    return buf

def img_file_to_gif(imgs_array, output_file_name):
    imageio.mimsave(output_file_name, imgs_array, duration=0.5)

def show_mixup_image(img1: np.array, img2: np.array, img3: np.array,
                     label1: int, label2: int, lam: float,
                     pca1: np.array, pca2: np.array, pca3: np.array, pca: np.array):
    """Visulaize virtual data and check the data is in vicinity distribution."""
    # subplot 4
    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(img1)
    ax[0].set_title(label="img1 {}".format(label1))
    ax[1].imshow(img2)
    ax[1].set_title(label="img2 {}".format(label2))
    ax[2].imshow(img3 / 255)
    ax[2].set_title(label="lam:{}".format(lam))

    ax[3].scatter(pca1[0], pca1[1], color='red')
    ax[3].annotate(label1, xy=pca1)
    ax[3].scatter(pca2[0], pca2[1], color='blue')
    ax[3].annotate(label2, xy=pca2)
    ax[3].scatter(pca3[0], pca3[1], color='purple')
    ax[3].annotate('mixup', xy=pca3)

    verts = [pca1, pca2]
    codes = [Path.MOVETO, Path.MOVETO]

    line_path = Path(verts, codes)
    patch = patches.PathPatch(line_path, facecolor='none', lw=2)
    pca_x = []
    pca_y = []
    for x, y in pca:
        pca_x.append(x)
        pca_y.append(y)
    ax[3].scatter(pca_x, pca_y, color='gray', alpha=0.3)
    ax[3].set_title(label="PCA_with lambda {}".format(lam))
    ax[3].add_patch(patch)
    xs, ys = zip(*verts)
    ax[3].plot(xs, ys, 'x--', lw= 2, color="black", alpha=0.4)
    return fig2data(fig)

if __name__ == "__main__":
    # (x, y), _ = mnist.load_data()
    (x, y), _ = cifar10.load_data()
    # (x, y), _ = fashion_mnist.load_data()
    x_flatten = np.reshape(x, (len(x), -1))
    random_idx1 = list(np.random.randint(0, len(x) - 1, 16))
    random_idx2 = list(np.random.randint(0, len(x) - 1, 16))

   # x1, y1 pairs
    x1 = x[random_idx1]
    y1 = y[random_idx1]

    # x2, y2 pairs
    x2 = x[random_idx2]
    y2 = y[random_idx2]
    _x, y_a, y_b, lam = mixup_data(x, y, alpha=hyper_params['alpha'])
    _x = np.reshape(_x, (len(_x), -1))

    x_flatten = np.concatenate([x_flatten, _x], axis=0)
    pca = PCA(n_components=2)
    pca.fit(x_flatten)

    principal_components = pca.transform(x_flatten)
    total = principal_components[random_idx1 + random_idx2]
    lam = np.array(range(1, 10)) * 0.1

    for idx1, idx2 in zip(random_idx1, random_idx2):
        x1 = x[idx1]
        x2 = x[idx2]
        y1 = y[idx1]
        y2 = y[idx2]
        figures = []
        path = './results/sample______{}_{}.gif'.format(y1, y2)
        for i, l in enumerate(lam): # lambda
            mixed_x, y1, y2, l = mixup_data_one_sample(x1, y1, x2, y2, l)
            x1_flatten = np.reshape(x1, (1, -1))
            x2_flatten = np.reshape(x2, (1, -1))
            mixed_x_flatten = np.reshape(mixed_x, (1, -1))
            pca1 = pca.transform(x1_flatten)[0]
            pca2 = pca.transform(x2_flatten)[0]
            pca3 = pca.transform(mixed_x_flatten)[0]
            figures.append(show_mixup_image(x1, x2, mixed_x,
                             y1, y2, l,
                             pca1, pca2, pca3, total))

        img_file_to_gif(figures, path)
        break







