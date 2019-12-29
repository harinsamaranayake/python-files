# https://stackoverflow.com/questions/30145957/plotting-2d-kernel-density-estimation-with-python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as pl

Y = np.random.multivariate_normal((0, 0), [[0.8, 0.05], [0.05, 0.7]], 100)
ax = sns.kdeplot(Y, shade = True, cmap = "PuBu")
ax.patch.set_facecolor('white')
ax.collections[0].set_alpha(0)
ax.set_xlabel('$Y_1$', fontsize = 15)
ax.set_ylabel('$Y_0$', fontsize = 15)
pl.xlim(-3, 3)
pl.ylim(-3, 3)
pl.plot([-3, 3], [-3, 3], color = "black", linewidth = 1)
pl.show()