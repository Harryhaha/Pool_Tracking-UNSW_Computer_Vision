import sys
import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt


def color_HSV_distribution(img_path):
	image = cv2.imread(img_path, -1)
	image = cv2.medianBlur(image,3)
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	cv2.imshow('pic',image)

	color = ('r','g','b')
	for channel,col in enumerate(color):
		histr = cv2.calcHist([hsv],[channel],None,[256],[0,256])

		max_index = np.argmax(histr)
		print(max_index)

		plt.plot(histr,color = col)
		plt.xlim([0,256])
	plt.title('Histogram for color scale picture')
	plt.show()

	while True:
		k = cv2.waitKey(0) & 0xFF	 
		if k == 27: break			 # ESC key to exit 
	cv2.destroyAllWindows()

if __name__ == '__main__':
	# args handle
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", help="path to the image")
	args = vars(ap.parse_args())
	image_path = args["image"]

	#analyze color
	color_HSV_distribution(image_path)
