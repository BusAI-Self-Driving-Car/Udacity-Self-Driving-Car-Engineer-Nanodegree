import os
import cv2

input_imgs = os.listdir(".")

for image in input_imgs: 
	img = cv2.imread(image, cv2.IMREAD_COLOR)
	resized_img = cv2.resize(img, (32, 32))
	cv2.imwrite("cropped/"+image, resized_img)