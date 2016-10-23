## find chessboard coners
FindChessboardCorners
https://www.reddit.com/r/computervision/comments/29t7bp/cvchess_automatically_inferring_chess_moves_from/


python3 colorAnalyze.py -i test_data/game1/balls/ball2.png

## detect red ball:
https://solarianprogrammer.com/2015/05/08/detect-red-circles-image-using-opencv/
there are two range

ball3: 
hsv range  ([0, 100, 150], [10, 200, 255])

process:
for each frame:
	blob -> image with keypoits

	for each ball:
		use range to find coutours
		find max coutours
		draw and record


## ball detection
The main goal in this project is to detect and identify balls. There are different approaches
to solving this problem. One approach is to concentrate on solving both problems at once by
identifying balls directly and thereby also implicitly detecting them. Other approaches, like the
one used in this project, performs the detection first to determine the regions of interest and then
extracts further data from the regions for use in the identification process.
The advantage of the former is that the computation can be done in one pass. Features that are
used to identify the balls, can in this way also be used in the detection process. This results in a
more robust detection because that a correctly identified ball implies a correct detection.
The advantage of the latter is that the identification can compare the detected regions against
each other, and thereby use this knowledge to, e.g. not include two of the same balls.

## issue
ball 0 not found on frame 41!
ball 0 not found on frame 57!
ball 0 not found on frame 58!
ball 0 not found on frame 69!

## mask
~~~
table_lower = self.table.hsv_color_lower
table_upper = self.table.hsv_color_upper

frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask_remove_table = cv2.inRange(frame_hsv, table_lower, table_upper)
mask_remove_table = cv2.bitwise_not(mask_remove_table)

mask_white = 

test_img = cv2.bitwise_and(frame, frame, mask=mask_remove_table)
cv2.imshow("masked", test_img)
cv2.waitKey(0)

mask_roi = np.zeros((frame.shape[0], frame.shape[1]), dtype=frame.dtype)
                    cv2.circle(mask_roi, (x, y), radius, 1, thickness=-1)
~~~

## issue
balls_data = {
    "0":
        {"hsv_color_lower": (0, 0, 100),
         "hsv_color_upper": (40, 80, 255),
         "trajectory_color": (255, 255, 255)
         },
    "3":
        {"hsv_color_lower": (5,170,170),
         "hsv_color_upper": (20,255,255),
         "trajectory_color": (134, 0, 252)
         },
    "2":
        {"hsv_color_lower": (99, 30, 50),
         "hsv_color_upper": (119, 200, 150),
         "trajectory_color": (112, 77, 49)
         }
}
