import cv2

image_size = (256, 256)  # (true_img_width, true_img_length)

# image_name = 'img_000001942.png'
# image_path = '/Users/harinsamaranayake/Desktop/Results_Latest/unet-master-new-256/RESULTS/EP10000/BOTH/' + image_name
# save_path = '/Users/harinsamaranayake/Desktop/' + image_name

image_path = '/Users/harinsamaranayake/Downloads/onr/img_000000101-1000.png'
save_path = image_path + '_THRESH' + '.png'

img = cv2.imread(image_path)  # grayscale
# img = cv2.resize(img, image_size)
ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# cv2.imshow(image_name, img)
cv2.waitKey(0)
cv2.imwrite(save_path, img)

print('Image Saved')
