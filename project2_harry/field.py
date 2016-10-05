import numpy as np
import numpy.linalg as la
import cv2
import math
import matplotlib.pyplot as plt
import time

# Count runtime
start_time = time.time()

filename_topview = 'test_data/field_img/top-view.jpg'
filename_sideview = 'test_data/field_img/side-view.jpg'

field_corners = np.empty([4,2], dtype = "float32")
field_counter = 0

tracking_point = [0 , 0]


def field_click(event, x, y, flags, param):
	global field_counter
	global tracking_point

	if event == cv2.EVENT_LBUTTONDBLCLK:
		if field_counter < 4:
			field_corners[field_counter, :] = [x,y]
			print (x,y)
			field_counter +=1
		elif field_counter == 4:
			print ("tracking point:")
			print (x,y)
			tracking_point=[x,y]
		else:
			print ("Press any key to continue")

def create_homography():
	global field_counter
	global filename_topview
	global filename_sideview

	hgcoord_filepath = 'hgmatrix.txt'

	top_image = cv2.imread(filename_topview) #not used
	side_image = cv2.imread(filename_sideview)

	print ("Select the four corners from the Background")
	print ("The corners should be selected: Left-Down, Left-Top, Right-Top, Right-Down")
	cv2.namedWindow('Side-View')
	cv2.setMouseCallback('Side-View', field_click)
	cv2.imshow('Side-View', side_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	side_view_corners = np.copy(field_corners)

	top_view_corners = np.array([[44, 393], [44, 30], [598,30], [598, 393]], dtype  = "float32")

	H = cv2.findHomography(side_view_corners, top_view_corners)[0]
	np.savetxt(hgcoord_filepath, H)
	return H

def create_topview(hg_matrix, input_pts):
	global filename_topview

	top_image = cv2.imread(filename_topview)

	pts = np.matrix(np.zeros(shape=(len(input_pts),3)))
	c = 0
	for i in input_pts:
		x,y = i[0][0], i[0][1]
		pts[c,:] = np.array([x,y,1], dtype = "float32")
		c+=1
	player_top_points = list()
	newPoints = np.empty([len(input_pts),3], dtype = "float32")
	c = 0
	for i in pts:
		newPoints = hg_matrix*(i.T)
		x = int(newPoints[0]/float(newPoints[2]))
		y = int(newPoints[1]/float(newPoints[2]))
		if(input_pts[c][1][0] == 'r'):
			cv2.circle(top_image,(x,y),3,(0,0,255),-1)
		elif(input_pts[c][1][0] == 'b'):
			cv2.circle(top_image,(x,y),3,(255,0,0),-1)
		else:
			cv2.circle(top_image,(x,y),3,(255,255,255),-1)
		player_top_points.append([[x, y], input_pts[c][1][0]])
		c +=1
	return top_image, player_top_points

if __name__ == '__main__':
	# global filename_topview

	H = create_homography()
	# p1 = np.array([[599,243]], dtype='float32')
	# p1 = np.array([p1])
	p1 = np.array([tracking_point], dtype='float32')
	p1 = np.array([p1])
	p_result = cv2.perspectiveTransform(p1, H)
	print("point: ")
	print(p_result[0][0])

	#draw
	top_image = cv2.imread(filename_topview)
	cv2.circle(top_image,(p_result[0][0][0],p_result[0][0][1]), 5, (0,0,255), -1)
	cv2.imshow('Top-View', top_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
