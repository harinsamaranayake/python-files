import numpy as np
import cv2
import matplotlib.pyplot as plt

image_name = "img_000000101.png"
image_path = "/Users/harinsamaranayake/Documents/Research/Datasets/FCN8s/split/on_road_test_color/"+image_name
img = cv2.imread(image_path)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)

# h = hsv[:,:,0]
# cv2.imshow('h',h)
# cv2.waitKey(0)

vectorized = img.reshape((-1,3))
vectorized = np.float32(vectorized)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
attempts=10

ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)

# print(vectorized.shape,'\t',ret,'\t',label.shape,'\t',center.shape)
print(label,'\n',center,'\n')

res = center[label.flatten()]
result_image = res.reshape((img.shape))

print(res,'\n')

plt.figure(figsize=(15,10))
plt.subplot(1,2,1),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(result_image)
plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
plt.show()