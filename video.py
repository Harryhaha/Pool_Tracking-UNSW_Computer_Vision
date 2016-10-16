import numpy as np
import cv2
from collections import deque, defaultdict

import imutils
import config


class VideoTable:
    def __init__(self, img_path, auto_find_corner=False):
        """
        :param img_path: video table image.
        :param auto_find_corner:
        """

        self.table_img_path = img_path

        # define the pool table color range
        self.hsv_color_lower = np.array([95, 100, 0], dtype="uint8")
        self.hsv_color_upper = np.array([105, 255, 255], dtype="uint8")

        self.field_corners = np.empty([4, 2], dtype="float32")  # order: Left-Down, Left-Top, Right-Top, Right-Down
        self.click_count = 0

        if auto_find_corner:
            pass
        else:
            self.manual_table_corner_detection()

    def event_table_corner_click(self, event, x, y, flags, param):
        # print("hello")

        # if event == cv2.EVENT_LBUTTONDOWN:
        if event == cv2.EVENT_LBUTTONDBLCLK:
            if self.click_count < 4:
                self.field_corners[self.click_count, :] = [int(x), int(y)]
                print(x, y)
                self.click_count += 1
            else:
                print("please close image window")

    def manual_table_corner_detection(self):

        field_corners = np.empty([4, 2], dtype="float32")
        field_counter = 0

        side_image = cv2.imread(self.table_img_path)

        print("Please select four corners from the pool table!")
        print("The corners should be selected: Left-Down, Left-Top, Right-Top, Right-Down")
        cv2.imshow('Side-View', side_image)
        cv2.setMouseCallback('Side-View', self.event_table_corner_click)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        # todo: auto table corner detection
        # def auto_pool_table_detection():
        #     side_image = cv2.imread(filename_sideview)
        #     side_image = cv2.medianBlur(side_image, 5)
        #
        #     height, width, channels = side_image.shape
        #
        #     # image = cv2.medianBlur(image,3)
        #     hsv_image = cv2.cvtColor(side_image, cv2.COLOR_BGR2HSV)
        #
        #     # define the color range
        #     lower = np.array([95, 100, 0], dtype="uint8")
        #     upper = np.array([105, 255, 255], dtype="uint8")
        #
        #     mask = cv2.inRange(hsv_image, lower, upper)
        #
        #     # show the masked output
        #     output = cv2.bitwise_and(side_image, side_image, mask=mask)
        #     # cv2.imshow("maskedEffect", np.hstack([image, output]))
        #     # cv2.imshow("maskedEffect", output)
        #     # cv2.imshow("Mask", mask)
        #
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        #
        #     # Set up the detector with default parameters.
        #
        #     # Setup SimpleBlobDetector parameters.
        #     params = cv2.SimpleBlobDetector_Params()
        #     # Filter by Convexity
        #     params.filterByConvexity = True
        #     params.minConvexity = 0.2
        #
        #     detector = cv2.SimpleBlobDetector_create(params)
        #
        #     # Detect blobs.
        #     keypoints = detector.detect(output)
        #
        #     # Draw detected blobs as red circles.
        #     # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        #     # im_with_keypoints = cv2.drawKeypoints(output, keypoints, np.array([]), (0, 0, 255),
        #     #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #
        #     # draw convexHull
        #     contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        #
        #     table_contours = max(contours, key=cv2.contourArea)
        #     hull = cv2.convexHull(table_contours)
        #     # print(hull)
        #
        #     # cv2.drawContours(output, [hull], -1, (0, 255, 0), 2)
        #     # cv2.imshow("table hull", output)
        #     # cv2.waitKey(0)
        #     # cv2.destroyAllWindows()
        #
        #     # pure_table_contours_image = np.zeros((height, width, 1), np.uint8)
        #     # cv2.drawContours(pure_table_contours_image, [hull], -1, 255, -1)  # thickness negative -> fill; positive means thick
        #     # cv2.imshow("hull", pure_table_contours_image)
        #     # cv2.waitKey(0)
        #     # cv2.destroyAllWindows()
        #
        #     '''Find Corner of Table'''
        #     test = cv2.approxPolyDP(hull, epsilon=40, closed=True)
        #     # print(test)
        #
        #     test_img = np.zeros((height, width, 1), np.uint8)
        #     cv2.drawContours(test_img, [test], -1, 255, -1)  # thickness negative -> fill; positive means thick
        #     cv2.imshow("appox", test_img)
        #
        #     points = test.tolist()
        #     # print(points)
        #     tmp_corner_points = []
        #     for p in points:
        #         print(p)
        #         x, y = p[0]
        #         cv2.circle(side_image, (x, y), 5, (0, 0, 255), -1)
        #
        #         tmp_corner_points.append((x, y))
        #
        #     print(tmp_corner_points)
        #
        #     # odered from p0-3
        #     # determine which point is on left-up (near 0,0)
        #     min_distance = float("inf")
        #     tmp_p1 = None
        #     for p in tmp_corner_points:
        #         x, y = p
        #         distance = ((x - 0) ^ 2 + (y - 0) ^ 2)
        #         if distance < min_distance:
        #             min_distance = distance
        #             tmp_p1 = [x, y]
        #
        #     field_corners[1, :] = tmp_p1
        #
        #     for p in tmp_corner_points:
        #         if tmp_p1[0] == p[0] and tmp_p1[1] == p[1]:
        #             continue
        #
        #         if p[0] - tmp_p1[0] > 200 and p[1] - tmp_p1[1] > 200:
        #             print("3:" + str(p))
        #             field_corners[3, :] = p
        #         elif p[0] - tmp_p1[0] > 200:
        #             print("2:" + str(p))
        #             field_corners[2, :] = p
        #         else:
        #             print("0:" + str(p))
        #             field_corners[0, :] = p
        #
        #     print("auto corner:")
        #     print(field_corners)
        #
        #     cv2.imshow("appox1", side_image)
        #
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()


class VideoBall:
    def __init__(self, ball_id, data=None):
        self.ball_id = ball_id
        self.hsv_color_lower = None
        self.hsv_color_lower = None

        if data:
            self.hsv_color_lower = data["hsv_color_lower"]
            self.hsv_color_upper = data["hsv_color_upper"]


class Video:
    def __init__(self, video_file):
        self.video_file = video_file
        # self.camera = cv2.VideoCapture(video_file)

        self.balls = {}
        self.table = None
        self.ball_tracking_rec_for_real_time = defaultdict(deque)  # limited space, tmp
        self.ball_tracking_rec_for_sim = {}  # unlimited space, for post analysis

        self.init_table()
        self.init_balls()

    def init_table(self):
        self.table = VideoTable("test_data/check0.png")

    def init_balls(self):
        for ball_id in config.balls_data:
            self.balls[ball_id] = VideoBall(ball_id, data=config.balls_data[ball_id])
            self.ball_tracking_rec_for_real_time[ball_id] = deque(maxlen=64)
            self.ball_tracking_rec_for_sim[ball_id] = []

    def detect_ball_from_frame(self, frame, frame_count, is_draw=True):
        # resize the frame, blur it, and convert it to the HSV color space
        frame = imutils.resize(frame, width=800)
        # frame = cv2.GaussianBlur(frame, (11, 11), 0)

        image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        """
        Simple Blob Finding interesting area
        - use table color range, generate image that only display table area
        - apply blob detection, find key points
        - generate image that only display interesting area. (potential ball areas)
        """
        # define the pool table color range
        table_lower = self.table.hsv_color_lower
        table_upper = self.table.hsv_color_upper

        mask_table_blue = cv2.inRange(image_hsv, table_lower, table_upper)
        image_with_table_area = cv2.bitwise_and(frame, frame, mask=mask_table_blue)
        # cv2.imshow("filtered pool table image", image_with_table_area)
        # cv2.waitKey(0)

        # Setup SimpleBlobDetector parameters. Using blob to morphology detect potential pool balls
        params = cv2.SimpleBlobDetector_Params()
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.2
        detector = cv2.SimpleBlobDetector_create(params)  # Initialize configured blob detector
        keypoints = detector.detect(image_with_table_area)

        # generate image with interesting area (areas which might have balls in it.)
        interesting_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=frame.dtype)
        for keypoint in keypoints:
            cv2.circle(interesting_mask, (int(keypoint.pt[0]), int(keypoint.pt[1])), int(keypoint.size / 2),
                       1, thickness=-1)

        img_with_interesting_area = cv2.bitwise_and(frame, frame, mask=interesting_mask)
        img_with_interesting_area_hsv = cv2.cvtColor(img_with_interesting_area, cv2.COLOR_BGR2HSV)
        # cv2.imshow("img_with_interesting_area", img_with_interesting_area)
        # cv2.waitKey(0)

        # filter out table background
        mask_table = cv2.inRange(img_with_interesting_area_hsv, table_lower, table_upper)
        mask_table = cv2.bitwise_not(mask_table)
        img_with_interesting_area = cv2.bitwise_and(img_with_interesting_area, img_with_interesting_area, mask=mask_table)
        # cv2.imshow("img_with_interesting_area (filter out table background)", img_with_interesting_area)
        # cv2.waitKey(0)
        img_with_interesting_area_hsv = cv2.cvtColor(img_with_interesting_area, cv2.COLOR_BGR2HSV)

        """
        detect balls using ball color range
        """
        for ball_id in self.balls:
            # print("ball_id", ball_id)

            """
            apply ball color range on interesting area
            """
            ball_color_lower = np.array(self.balls[ball_id].hsv_color_lower, dtype="uint8")
            ball_color_upper = np.array(self.balls[ball_id].hsv_color_upper, dtype="uint8")
            mask_range = cv2.inRange(img_with_interesting_area_hsv, ball_color_lower, ball_color_upper)
            mask_range = cv2.erode(mask_range, None, iterations=2)
            mask_range = cv2.dilate(mask_range, None, iterations=2)

            # only display specific ball
            image_with_ball = cv2.bitwise_and(img_with_interesting_area, img_with_interesting_area,
                                              mask=mask_range)
            # cv2.imshow(ball_id, image_with_ball)
            # cv2.waitKey(0)

            """
            find cnts
            """
            cnts = cv2.findContours(mask_range.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            if len(cnts) <= 0:
                # print("ball", ball_id, "not found!")
                print("ball {} not found on frame {} !".format(ball_id, frame_count))
                # self.ball_tracking_rec_for_real_time[ball_id].append(None) # no need
                self.ball_tracking_rec_for_sim[ball_id].append(None)
                continue

            # print("ball", ball_id, "found!")
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)

            ball_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            self.ball_tracking_rec_for_real_time[ball_id].appendleft(tuple(ball_center))
            self.ball_tracking_rec_for_sim[ball_id].append(tuple(ball_center))

            """
            draw balls and trajectory
            """
            if is_draw:
                # cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 0), 2)
                cv2.circle(frame, ball_center, 4, (255, 255, 255), 2)

                # trajectory
                tmp_ball_record = self.ball_tracking_rec_for_real_time[ball_id]
                # print(tmp_ball_record)
                for i in range(1, len(tmp_ball_record)):
                    if tmp_ball_record[i - 1] is None or tmp_ball_record[i] is None:
                        # print("Doesn't draw the line for current frame!")
                        continue

                    # thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                    thickness = int(np.sqrt(64 / float(i + 1)) * 1.2)

                    # print(thickness)
                    # print(tmp_ball_record[i - 1])
                    # print(tmp_ball_record[i])

                    cv2.line(frame, tmp_ball_record[i - 1], tmp_ball_record[i], (255, 255, 255), thickness)

        """
        after check every ball, show processed frame
        """
        # print("tracking record:", self.ball_tracking_rec_for_real_time)

        # show the frame to our screen
        cv2.imshow("Pool ball tracking frame", frame)

    def real_time_tracking(self):
        camera = cv2.VideoCapture(self.video_file)
        frame_count = 0
        while True:
            frame_count += 1

            # grab the current frame
            (grabbed, frame) = camera.read()

            # if we are viewing a video and we did not grab a frame,
            # then we have reached the end of the video
            if self.video_file and not grabbed:
                break

            self.detect_ball_from_frame(frame,frame_count)
            # print(self.ball_tracking_rec_for_real_time)

            key = cv2.waitKey(1) & 0xFF
            # if the 'q' key is pressed, stop the loop
            if key == ord("q"):
                break

        camera.release()

if __name__ == '__main__':
    # video_table = VideoTable("test_data/check0.png")
    # print(video_table.field_corners)

    video_file = "test_data/game1/video/1.mp4"

    myvideo1 = Video(video_file)
    myvideo1.real_time_tracking()




