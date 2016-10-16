import numpy as np
import numpy.linalg as la
import cv2
import math
import matplotlib.pyplot as plt
import time

# Count runtime
start_time = time.time()

filename_topview = 'test_data/pool_topview/topview1.gif'
filename_sideview = 'test_data/game1/1.png'

field_corners = np.empty([4, 2], dtype="float32")  # order: Left-Down, Left-Top, Right-Top, Right-Down
field_counter = 0

# plain_pool_table_corners = np.empty([4,2], dtype = "float32")
# plain_pool_table_click_counter = 0;

tracking_point = []


# def plain_pool_table_corner_click(event, x, y, flags, param):
# 	global plain_pool_table_click_counter
# 	global plain_pool_table_corners

# 	if event == cv2.EVENT_LBUTTONDBLCLK:
# 		if field_counter < 4:
# 			plain_pool_table_corners[field_counter, :] = [x,y]
# 			print (x,y)
# 			plain_pool_table_click_counter +=1
# 		else:
# 			print ("Press any key to continue")

def auto_pool_table_detection():
    side_image = cv2.imread(filename_sideview)
    side_image = cv2.medianBlur(side_image, 5)
    height, width, channels = side_image.shape

    # image = cv2.medianBlur(image,3)
    hsv_image = cv2.cvtColor(side_image, cv2.COLOR_BGR2HSV)

    # define the color range
    lower = np.array([95, 100, 0], dtype="uint8")
    upper = np.array([105, 255, 255], dtype="uint8")

    mask = cv2.inRange(hsv_image, lower, upper)

    # show the masked output
    output = cv2.bitwise_and(side_image, side_image, mask=mask)
    # cv2.imshow("maskedEffect", np.hstack([image, output]))
    cv2.imshow("maskedEffect", output)
    cv2.imshow("Mask", mask)
    cv2.imwrite("1.jpg",mask)

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

    #=================Detect blobs===================
    # Detect blobs.
    keypoints = detector.detect(output)
    # keypoints[0]: pt->center, size->radius
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(output, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.imwrite("2.jpg",im_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #=======================/Detect blobs==============

    # draw convexHull
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    # print(contours)

    table_contours = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(table_contours)
    # test hull
    print(hull)
    print("=====hull end=====")

    cv2.drawContours(output, [hull], -1, (0, 255, 0), 2)
    cv2.imshow("table hull", output)
    cv2.imwrite("3.jpg",output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pure_table_contours_image = np.zeros((height, width, 1), np.uint8)
    cv2.drawContours(pure_table_contours_image, [hull], -1, 255, -1)  # thickness negative -> fill; positive means thick
    cv2.imshow("hull", pure_table_contours_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    '''Find Corner of Table'''
    test = cv2.approxPolyDP(hull, epsilon=40, closed=True)
    # print(test)

    test_img = np.zeros((height, width, 1), np.uint8)
    cv2.drawContours(test_img, [test], -1, 255, -1)  # thickness negative -> fill; positive means thick
    cv2.imshow("appox", test_img)
    cv2.imwrite("4.jpg",test_img)

    points = test.tolist()
    # print(points)
    tmp_corner_points = []
    for p in points:
        print(p)
        x, y = p[0]
        cv2.circle(side_image, (x, y), 5, (0, 0, 255), -1)

        tmp_corner_points.append((x, y))

    print(tmp_corner_points)

    # odered from p0-3
    # determine which point is on left-up (near 0,0)
    min_distance = float("inf")
    tmp_p1 = None
    for p in tmp_corner_points:
        x, y = p
        distance = ((x - 0) ^ 2 + (y - 0) ^ 2)
        if distance < min_distance:
            min_distance = distance
            tmp_p1 = [x, y]

    field_corners[1, :] = tmp_p1

    for p in tmp_corner_points:
        if tmp_p1[0] == p[0] and tmp_p1[1] == p[1]:
            continue

        if p[0] - tmp_p1[0] > 200 and p[1] - tmp_p1[1] > 200:
            print("3:" + str(p))
            field_corners[3, :] = p
        elif p[0] - tmp_p1[0] > 200:
            print("2:" + str(p))
            field_corners[2, :] = p
        else:
            print("0:" + str(p))
            field_corners[0, :] = p

    print("auto corner:")
    print(field_corners)

    cv2.imshow("appox1", side_image)
    cv2.imwrite("5.jpg",side_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    '''
    # edges = cv2.Canny(table_image_gray,50,150,apertureSize = 3)
    # cv2.imshow("test", edges)
    # lines = cv2.HoughLinesP(edges, 1, np.pi/180,1)
    # print(lines)
    # hough_img = np.zeros((height,width,3), np.uint8)
    # for x1,y1,x2,y2 in lines[0]:
    # 	cv2.line(hough_img, (x1,y1), (x2,y2), (0,255,0), 100)

    # cv2.imshow("hough lines", hough_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    '''


def click_tracking_point(event, x, y, flags, param):
    global tracking_point

    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("tracking point: " + str(x) + "," + str(y))
        tracking_point.append([x, y])


def click_manual_table_corner(event, x, y, flags, param):
    global field_counter
    global tracking_point

    if event == cv2.EVENT_LBUTTONDBLCLK:
        if field_counter < 4:
            field_corners[field_counter, :] = [int(x), int(y)]
            print(x, y)
            field_counter += 1
        else:
            print("please close image window")


def manual_table_corner_detection():
    global field_counter
    global field_corners
    global tracking_point

    field_corners = np.empty([4, 2], dtype="float32")
    field_counter = 0
    tracking_point = []

    side_image = cv2.imread(filename_sideview)

    print("Select the four corners from the Background")
    print("The corners should be selected: Left-Down, Left-Top, Right-Top, Right-Down")
    cv2.imshow('Side-View', side_image)
    cv2.setMouseCallback('Side-View', click_manual_table_corner)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_homography():
    global field_counter
    global field_corners
    global filename_topview
    global filename_sideview

    hgcoord_filepath = 'hgmatrix.txt'

    # top_image = cv2.imread(filename_topview)

    # print ("Select the four corners of 2D pool table")
    # print ("The corners should be selected: Left-Down, Left-Top, Right-Top, Right-Down")
    # cv2.imshow('Top-View', top_image)
    # cv2.setMouseCallback('Top-View', plain_pool_table_corner_click)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # top_view_corners = np.array([[44, 393], [44, 30], [598,30], [598, 393]], dtype  = "float32")
    top_view_corners = np.array([[42, 611], [42, 36], [975, 36], [975, 611]], dtype="float32")
    # top_view_corners = np.copy(plain_pool_table_corners)



    # manual_table_corner_detection()
    # if field_counter<4:
    # 	print("You need to click 4 corners!! Try again!")
    # 	exit(0)

    auto_pool_table_detection()
    # print(field_corners)

    # click for tracking points
    side_image = cv2.imread(filename_sideview)
    print("Click for tracking points!")
    cv2.imshow('Side-View', side_image)
    cv2.setMouseCallback('Side-View', click_tracking_point)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    side_view_corners = np.copy(field_corners)

    H = cv2.findHomography(side_view_corners, top_view_corners)[0]
    np.savetxt(hgcoord_filepath, H)
    return H


# def create_topview(hg_matrix, input_pts):
# 	global filename_topview

# 	top_image = cv2.imread(filename_topview)

# 	pts = np.matrix(np.zeros(shape=(len(input_pts),3)))
# 	c = 0
# 	for i in input_pts:
# 		x,y = i[0][0], i[0][1]
# 		pts[c,:] = np.array([x,y,1], dtype = "float32")
# 		c+=1
# 	player_top_points = list()
# 	newPoints = np.empty([len(input_pts),3], dtype = "float32")
# 	c = 0
# 	for i in pts:
# 		newPoints = hg_matrix*(i.T)
# 		x = int(newPoints[0]/float(newPoints[2]))
# 		y = int(newPoints[1]/float(newPoints[2]))
# 		if(input_pts[c][1][0] == 'r'):
# 			cv2.circle(top_image,(x,y),3,(0,0,255),-1)
# 		elif(input_pts[c][1][0] == 'b'):
# 			cv2.circle(top_image,(x,y),3,(255,0,0),-1)
# 		else:
# 			cv2.circle(top_image,(x,y),3,(255,255,255),-1)
# 		player_top_points.append([[x, y], input_pts[c][1][0]])
# 		c +=1
# 	return top_image, player_top_points

if __name__ == '__main__':
    # global filename_topview

    H = create_homography()
    # p1 = np.array([[599,243]], dtype='float32')
    # p1 = np.array([p1])

    top_image = cv2.imread(filename_topview)
    side_image = cv2.imread(filename_sideview)

    for tp1 in tracking_point:
        p1 = np.array([tp1], dtype='float32')
        p1 = np.array([p1])
        p_result = cv2.perspectiveTransform(p1, H)
        print("point: ")
        print(p_result[0][0])

        # draw
        cv2.circle(side_image, (tp1[0], tp1[1]), 5, (0, 0, 255), -1)

        cv2.circle(top_image, (p_result[0][0][0], p_result[0][0][1]), 5, (0, 0, 255), -1)

    # display
    cv2.imshow('Side-View', side_image)
    cv2.imwrite("video_points.jpg",side_image)

    cv2.imshow('Top-View', top_image)
    cv2.imwrite("2D_points.jpg",top_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
