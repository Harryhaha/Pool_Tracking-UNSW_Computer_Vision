__author__ = "Harry"

import cv2
import sys
import imutils
from collections import deque
import numpy as np

pts = deque(maxlen=64)

video = sys.argv[1]
camera = cv2.VideoCapture(video)

writeLower = (10, 50, 0)
writeUpper = (50, 100, 255)


boundaries = [
		# ([10, 50, 0], [50, 100, 255]),  # write ball
		([0, 10, 0], [100, 100, 255]),  # write ball

		# ([20, 100, 0], [40, 255, 255]),  # ball1
        #
		# # ([110, 60, 0], [170, 255, 255]),  # ball2
		# ([110, 60, 0], [170, 255, 255]),  # ball2
        #
		# ([0, 0, 0], [255, 255, 50]),  # ball8
        #
		# ([0, 160, 0], [20, 255, 255]),  # ball3  # ([240, 100, 0], [255, 255, 255])
		# ([0, 100, 0], [10, 200, 255]),  # ball5
		# ([60, 90, 0], [90, 255, 255]),  # ball6
		# # ([0, 20, 0], [40, 90, 255]),      #ball7
]


# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if video and not grabbed:
        break

    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=500)
    # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    # Using blob to recognize pool balls, using color to identify specific ball
    # Currently only recogize write ball and try to track it
    table_lower = np.array([95, 100, 0], dtype="uint8") # define the pool table color range
    table_upper = np.array([105, 255, 255], dtype="uint8")
    mask_table_blue = cv2.inRange(image_hsv, table_lower, table_upper)
    table_image = cv2.bitwise_and(frame, frame, mask=mask_table_blue)
    # cv2.imshow("filtered pool table image", table_image)
    # cv2.waitKey(0)


    # Setup SimpleBlobDetector parameters. Using blob to morphology detect potential pool balls
    params = cv2.SimpleBlobDetector_Params()
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.2
    detector = cv2.SimpleBlobDetector_create(params) # Initialize blob

    keypoints = detector.detect(table_image)
    # keypoints is a collection of all blobs, they are potential pool balls
    # So check it in a loop
    for keypoint in keypoints:
        circle_img = np.zeros((frame.shape[0], frame.shape[1]), dtype=frame.dtype)
        cv2.circle(circle_img, (int(keypoint.pt[0]), int(keypoint.pt[1])), int(keypoint.size / 2), 1, thickness=-1)
        masked_img = cv2.bitwise_and(frame, frame, mask=circle_img)
        # cv2.imshow("harry previous image", masked_img)
        # masked_data = cv2.bitwise_not(masked_data, masked_data, mask=mask)
        masked_img_hsv = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)
        mask_new = cv2.inRange(masked_img_hsv, table_lower, table_upper)
        mask_new = cv2.bitwise_not(mask_new)
        # show the masked output
        masked_img = cv2.bitwise_and(masked_img, masked_img, mask=mask_new)
        # cv2.imshow("harry final image", masked_img)
        # cv2.waitKey(0)
        masked_img_hsv = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)
        # Currently only recognize write ball
        for (lower_range, upper_range) in boundaries:
            # create NumPy arrays from the boundaries
            lower_range = np.array(lower_range, dtype="uint8")
            upper_range = np.array(upper_range, dtype="uint8")
            mask_range = cv2.inRange(masked_img_hsv, lower_range, upper_range)
            mask_range = cv2.erode(mask_range, None, iterations=2)
            mask_range = cv2.dilate(mask_range, None, iterations=2)
            cnts = cv2.findContours(mask_range.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            center = None
            if len(cnts) > 0:
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # update the points queue
            pts.appendleft(center)

            # loop over the set of tracked points
            for i in range(1, len(pts)):
                # if either of the tracked points are None, ignore
                # them
                if pts[i - 1] is None or pts[i] is None:
                    continue

                # otherwise, compute the thickness of the line and
                # draw the connecting lines
                thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

            # show the frame to our screen
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the 'q' key is pressed, stop the loop
            if key == ord("q"):
                break



    ##########################################################################################
    ###Sample code from http://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/####
    ################# The code is to track tennis ball also using color range ################
    ##########################################################################################
    # # construct a mask for the color "green", then perform
    # # a series of dilations and erosions to remove any small
    # # blobs left in the mask
    # mask = cv2.inRange(hsv, writeLower, writeUpper)
    # mask = cv2.erode(mask, None, iterations=2)
    # mask = cv2.dilate(mask, None, iterations=2)
    #
    # # find contours in the mask and initialize the current
    # # (x, y) center of the ball
    # cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
    #                         cv2.CHAIN_APPROX_SIMPLE)[-2]
    # center = None
    #
    # # only proceed if at least one contour was found
    # if len(cnts) > 0:
    #     # find the largest contour in the mask, then use
    #     # it to compute the minimum enclosing circle and
    #     # centroid
    #     c = max(cnts, key=cv2.contourArea)
    #     ((x, y), radius) = cv2.minEnclosingCircle(c)
    #     M = cv2.moments(c)
    #     center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    #
    #     # only proceed if the radius meets a minimum size
    #     if radius > 10:
    #         # draw the circle and centroid on the frame,
    #         # then update the list of tracked points
    #         cv2.circle(frame, (int(x), int(y)), int(radius),
    #                    (0, 255, 255), 2)
    #         cv2.circle(frame, center, 5, (0, 0, 255), -1)
    #
    # # update the points queue
    # pts.appendleft(center)
    #
    # # loop over the set of tracked points
    # for i in range(1, len(pts)):
    #     # if either of the tracked points are None, ignore
    #     # them
    #     if pts[i - 1] is None or pts[i] is None:
    #         continue
    #
    #     # otherwise, compute the thickness of the line and
    #     # draw the connecting lines
    #     thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
    #     cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
    #
    # # show the frame to our screen
    # cv2.imshow("Frame", frame)
    # key = cv2.waitKey(1) & 0xFF
    #
    # # if the 'q' key is pressed, stop the loop
    # if key == ord("q"):
    #     break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

