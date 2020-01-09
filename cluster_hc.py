import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

X = np.array([[5, 3],
              [10, 15],
              [15, 12],
              [24, 10],
              [30, 30],
              [85, 70],
              [71, 80],
              [60, 78],
              [70, 55],
              [80, 91], ])

#..............Figure 01................#

labels = range(1, 11)
plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(X[:, 0], X[:, 1], label='True Position')

# set lables
for label, x, y in zip(labels, X[:, 0], X[:, 1]):
    plt.annotate(
        label,
        xy=(x, y),
        xytext=(-3, 3),
        textcoords='offset points', ha='right', va='bottom')

# plt.show()

#..............Figure 02................#

linked = linkage(X, 'average')

labelList = range(1, 11)

plt.figure(figsize=(10, 7))

dendrogram(linked,
           orientation='top',
           labels=labelList,
           distance_sort='descending',
           show_leaf_counts=True)

# plt.show()

# image_name = "img_000000101.png"
# image_path = "/Users/harinsamaranayake/Documents/Research/Datasets/FCN8s/split/on_road_test_color/"+image_name
# img = cv2.imread(image_path)
# img_gray = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
# cv2.imshow('img',img)
# cv2.waitKey(0)

# print(img)
# print(img.shape)
# img = img.reshape(img.shape[0]*img.shape[1],img.shape[2])

# print(img)
# print(img.shape)
# print(type(img))

# linked = linkage(img, 'average')

# print('done')

# plt.show()

#..............Figure 03................#

image_name = "img_000000101.png"
image_path = "/Users/harinsamaranayake/Documents/Research/Datasets/FCN8s/split/on_road_test_color/"+image_name
img = cv2.imread(image_path)
img = cv2.resize(img, (int(img.shape[1]/4), int(img.shape[0]/4))) 
#Note : 3 channels (90, 160, 3) 14400 if doubled RecursionError: maximum recursion depth exceeded while getting the str of an object
#Note : 1 channels (90, 160, 1) 14400 if doubled RecursionError: maximum recursion depth exceeded while getting the str of an object
print(img.shape)

# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = img.reshape((img.shape[0] * img.shape[1],1))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img[:, :, 0:1]
img = img.reshape((img.shape[0] * img.shape[1], 1))

print(img)
print(img.shape)

linked = linkage(img, 'average')

plt.figure(figsize=(10, 7))

dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)

plt.show()

print('done')
