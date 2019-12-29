
# https://medium.com/intel-student-ambassadors/kernel-density-estimation-with-python-using-sklearn-c50b3c337871import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits as ld
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture as GMM
from sklearn.mixture import BayesianGaussianMixture
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_digit_data(data):
    fig, ax = plt.subplots(12, 4, figsize=(8, 8),subplot_kw=dict(xticks=[], yticks=[]))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax.flat):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)

def kde(n):
    digit_data = ld()
    pca = PCA(n_components=n, whiten=False)
    data = pca.fit_transform(digit_data.data)
    params = {'bandwidth': np.logspace(-1, 1, 20)}
    grid = GridSearchCV(KernelDensity(), params, cv=5)
    grid.fit(data)
    print("bandwidth selcted : ",grid.best_estimator_.bandwidth )


    kde = grid.best_estimator_


    new_data = kde.sample(48, random_state=0)
    new_data = pca.inverse_transform(new_data)
    print()
    print("48 new data points generated : ")
    print()
    plot_digit_data(new_data)

kde(28)

plt.show()