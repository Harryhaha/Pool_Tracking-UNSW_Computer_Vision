# import the necessary packages
import numpy as np
import argparse
import cv2

def detect_ball_blob(img_path):
	raw_image = cv2.imread(img_path)
	raw_image = cv2.medianBlur(raw_image, 5)
	height, width, channels = raw_image.shape

	# image = cv2.medianBlur(image,3)
	hsv_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2HSV)

	# define the color range
	lower = np.array([95, 100, 0], dtype="uint8")
	upper = np.array([105, 255, 255], dtype="uint8")

	mask = cv2.inRange(hsv_image, lower, upper)

	# show the masked output
	output = cv2.bitwise_and(raw_image, raw_image, mask=mask)
	# cv2.imshow("maskedEffect", np.hstack([image, output]))
	cv2.imshow("maskedEffect", output)
	cv2.imshow("Mask", mask)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# Set up the detector with default parameters.

	# Setup SimpleBlobDetector parameters.
	params = cv2.SimpleBlobDetector_Params()
	# Filter by Convexity
	params.filterByConvexity = True
	params.minConvexity = 0.2

	# Filter by Area.
	params.filterByArea = True
	params.minArea = 100
	# params.maxArea = 100

	detector = cv2.SimpleBlobDetector_create(params)

	# =================Detect blobs===================
	# Detect blobs.
	keypoints = detector.detect(output)
	# keypoints[0]: pt->center, size->radius
	# Draw detected blobs as red circles.
	# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
	im_with_keypoints = cv2.drawKeypoints(output, keypoints, np.array([]), (0, 0, 255),
										  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	# Show keypoints
	cv2.imshow("Keypoints", im_with_keypoints)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	# =======================/Detect blobs==============

	# =====for each keypoints, use the blob area as mask to find candidate ball position





def detect_ball_via_color_mask(img_path):
	# load the image
	image = cv2.imread(img_path)
	# image = cv2.medianBlur(image,3)
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	# define the list of boundaries
	boundaries = [
		# ([10, 50, 0], [50, 100, 255]),  # cue
		# ([20, 100, 0], [40, 255, 255]),  # ball1
		# ([0, 0, 0], [255, 255, 50])  # ball8
		# ([2, 182, 170], [6, 222, 255]),  # ball13 primary_hue=4
		# ([0, 0, 150], [10, 200, 255])  # ball14 primary_hue=8
		# ([0, 0, 100], [50, 100, 255])  # ball0 primary_hue=25
		# ([50, 150, 20], [100, 255, 60]) # ball6 primary_hue=82
		([5, 170, 170], [20, 255, 255]) # ball6 primary_hue=82
	]

	# loop over the boundaries
	for (lower, upper) in boundaries:
		# create NumPy arrays from the boundaries
		lower = np.array(lower, dtype="uint8")
		upper = np.array(upper, dtype="uint8")

		# find the colors within the specified boundaries and apply
		# the mask
		# mask = cv2.inRange(image, lower, upper)
		mask = cv2.inRange(hsv, lower, upper)
		mask = cv2.erode(mask, None, iterations=2)
		mask = cv2.dilate(mask, None, iterations=2)

		# show the images
		output = cv2.bitwise_and(image, image, mask=mask)
		# cv2.imshow("maskedEffect", np.hstack([image, output]))
		cv2.imshow("maskedEffect", output)
		cv2.waitKey(0)

		# find contours in the mask and initialize the current
		# (x, y) center of the ball
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
								cv2.CHAIN_APPROX_SIMPLE)[-2]
		center = None

		# print(max(cnts, key=cv2.contourArea))
		# print("=====end=====")

		# only proceed if at least one contour was found
		if len(cnts) > 0:
			
			image_copy = image.copy()
			
			# we can find max match!!!
			# find the largest contour in the mask, then use
			# it to compute the minimum enclosing circle and
			# centroid
			# c = max(cnts, key=cv2.contourArea)
			

			for c in cnts:
				((x, y), radius) = cv2.minEnclosingCircle(c)

				# M = cv2.moments(c)
				# center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
				center = (int(x), int(y))

				# only proceed if the radius meets a minimum size
				radius = 20

				# draw the circle and centroid on the frame,
				# then update the list of tracked points
				
				cv2.circle(image_copy, (int(x), int(y)), int(radius),
						   (0, 255, 255), 2)
				cv2.circle(image_copy, center, 5, (0, 0, 255), -1)

			cv2.imshow("images", image_copy)
			cv2.waitKey(0)


if __name__ == '__main__':
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", help="path to the image")
	args = vars(ap.parse_args())

	image_path = args["image"]
	
	# detect_ball_blob(image_path)
	detect_ball_via_color_mask(image_path)
