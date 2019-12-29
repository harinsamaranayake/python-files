# https://www.youtube.com/watch?v=x5zLaWT5KPs
# https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
# https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
# https://scikit-learn.org/stable/modules/density.html
# https://pythonhosted.org/PyQt-Fit/KDE_tut.html

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

def make_data(N, f=0.3, rseed=1):
    rand = np.random.RandomState(rseed)
    x = rand.randn(N)
    x[int(f * N):] += 5
    # x[300:] += 5 Add 5 to all the values from 5 to end
    # print(rand)
    # print(x)
    return x

x = make_data(1000)

# Area under the histogram is equal to 1 - Normalized
# hist_normalized = plt.hist(x, bins=30, normed=True) 
# density, bins, patches = hist_normalized

# print(hist_normalized)
# print(density,'\n',bins,'\n',patches)

# bins - points where bins are created along the x axis
# widths = bins[1:] - bins[:-1] [ending point of a bin - starting point of a bin] Array substraction]
# Array substraction & multiplication
# widths = bins[1:] - bins[:-1]
# sum = (density * widths).sum()
# print(sum)

# https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html
x = make_data(20)
bins = np.linspace(-5, 10, 10)

fig, ax = plt.subplots(1, 2, figsize=(12, 4),
                       sharex=True, sharey=True,
                       subplot_kw={'xlim':(-4, 9),
                                   'ylim':(-0.02, 0.3)})
fig.subplots_adjust(wspace=0.05)

# 0,0.0 | 1,0.6 from enumarate 0.0 and 0.6 are off sets
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.full_like.html
# print(np.full_like(x, -0.01)) # [-0.01, ... , -0.01] - 20 items
# https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.plot.html
# |k |-marker shape k-black
# marking x - point = x | marking y - point = np.full_like(x, -0.01) > x[1],-0.01
for i, offset in enumerate([0.0, 0.6]):
    ax[i].hist(x, bins=bins + offset, normed=True)
    ax[i].plot(x, np.full_like(x, -0.01), '|k',markeredgewidth=1)

# ..........

# https://docs.scipy.org/doc/numpy/reference/generated/numpy.arange.html
fig, ax = plt.subplots()
bins = np.arange(-3, 8)
ax.plot(x, np.full_like(x, -0.1), '|k',markeredgewidth=1)

#https://realpython.com/python-zip-function/
# print(np.histogram(x, bins)) - (array([1, 1, 2, 1, 1, 1, 1, 6, 3, 3]), array([-3, -2, -1,  0,  1,  2,  3,  4,  5,  6,  7]))
# print(*np.histogram(x, bins)) - [1 1 2 1 1 1 1 6 3 3] [-3 -2 -1  0  1  2  3  4  5  6  7]
# zip - [1,-3]..[3,7]
# Rectangle(xy, width, height, angle=0.0, facecolor=None, color=None, linewidth=None, linestyle=None, antialiased=None, hatch=None, capstyle=None, joinstyle=None, **kwargs)
# for count, edge in zip(*np.histogram(x, bins)):
#     for i in range(count):
#         ax.add_patch(plt.Rectangle((edge, i), 1, 1,alpha=0.5))

ax.set_xlim(-4, 8)
ax.set_ylim(-0.2, 8)

# ..........

# x_d = np.linspace(-4, 8, 2000)
# density = sum((abs(xi - x_d) < 0.5) for xi in x) # check

# plt.fill_between(x_d, density, alpha=0.5)
# plt.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)

# plt.axis([-4, 8, -0.2, 8])

# ..........

from scipy.stats import norm
x_d = np.linspace(-4, 8, 2000)
density = sum(norm(xi).pdf(x_d) for xi in x) # check

plt.fill_between(x_d, density, alpha=0.5)
plt.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)

plt.axis([-4, 8, -0.2, 5])

plt.show()