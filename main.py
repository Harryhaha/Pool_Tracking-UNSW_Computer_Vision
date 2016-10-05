import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

fname = 'test_data/chessboard/c1.jpg'

img = cv2.imread(fname)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCorners(gray, (7,7),3)

print("=====corner=====")
print(corners)
print(len(corners))

corners2=cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)

print("=====corner2=====")
print(corners2)

# Draw and display the corners
cv2.drawChessboardCorners(img, (7,7), corners2, ret)
cv2.imshow('img',img)
cv2.waitKey(2000)
