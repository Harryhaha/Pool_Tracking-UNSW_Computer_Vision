## find chessboard coners
FindChessboardCorners
https://www.reddit.com/r/computervision/comments/29t7bp/cvchess_automatically_inferring_chess_moves_from/


python ballDetect.py --image pokemon_games.png

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
