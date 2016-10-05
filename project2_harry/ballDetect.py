# import the necessary packages
import numpy as np
import argparse
import cv2
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
# image = cv2.medianBlur(image,3)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# define the list of boundaries
boundaries = [
	([10, 50, 0], [50, 100, 255]),    	#write ball
	# ([10, 20, 0], [70, 100, 255]),  # write ball

	([20, 100, 0], [40, 255, 255]),   	#ball1
	([110, 70, 0], [170, 255, 255]),    #ball2
	([0, 0, 0], [255, 255, 50]),      	#ball8

	([0, 160, 0], [20, 255, 255]),      #ball3  # ([240, 100, 0], [255, 255, 255])
	([0, 100, 0], [10, 200, 255]),		#ball5
	([60, 90, 0], [90, 255, 255]),		#ball6
	# ([0, 20, 0], [40, 90, 255]),      #ball7
]

# loop over the boundaries
for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
 
	# find the colors within the specified boundaries and apply
	# the mask
	# mask = cv2.inRange(image, lower, upper)
	mask = cv2.inRange(hsv, lower, upper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	
	#show the images
	output = cv2.bitwise_and(image, image, mask = mask)
	# cv2.imshow("maskedEffect", np.hstack([image, output]))
	cv2.imshow("maskedEffect", output)
	cv2.waitKey(0)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None

	# print(max(cnts, key=cv2.contourArea))
	# print("=====end=====")
	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		
		# M = cv2.moments(c)
		# center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		center = (int(x),int(y))
 
		# only proceed if the radius meets a minimum size
		radius = 20

		# draw the circle and centroid on the frame,
		# then update the list of tracked points
		image_copy = image.copy()
		cv2.circle(image_copy, (int(x), int(y)), int(radius),
			(0, 255, 255), 2)
		cv2.circle(image_copy, center, 5, (0, 0, 255), -1)

		cv2.imshow("images", image_copy)
		cv2.waitKey(0)