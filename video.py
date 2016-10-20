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

        self.table_corners = np.empty([4, 2], dtype="float32")  # order: Left-Down, Left-Top, Right-Top, Right-Down
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
                self.table_corners[self.click_count, :] = [int(x), int(y)]
                print(x, y)
                self.click_count += 1
            else:
                print("please close image window")

    def manual_table_corner_detection(self):
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
            self.trajectory_color = data["trajectory_color"]


class Video:
    def __init__(self, video_file, resize=True):
        self.video_file = video_file
        self.resize = resize
        # self.camera = cv2.VideoCapture(video_file)
        self.first_frame = None

        self.balls = {}
        self.table = None
        self.tmp_ball_tracking_rec_for_trajectory = defaultdict(deque)  # limited space, tmp

        # unlimited space, for post analysis.
        # {"0":[(111,222),(111,333),None,...], "3":..., ...}
        self.ball_tracking_rec_complete = {}

        self.init_table()
        self.init_balls()

    def init_table(self):
        camera = cv2.VideoCapture(self.video_file)

        # use first frame to detect table
        (grabbed, self.first_frame) = camera.read()
        # cv2.imshow("first frame", self.first_frame)
        if self.resize:
            self.first_frame = imutils.resize(self.first_frame, width=800)

        cv2.imwrite(config.first_frame_save_path, self.first_frame)

        # self.table = VideoTable("test_data/check0.png")
        self.table = VideoTable(config.first_frame_save_path)

        camera.release()

    def init_balls(self):
        for ball_id in config.balls_data:
            self.balls[ball_id] = VideoBall(ball_id, data=config.balls_data[ball_id])
            self.tmp_ball_tracking_rec_for_trajectory[ball_id] = deque(maxlen=64)
            self.ball_tracking_rec_complete[ball_id] = []

    def get_img_with_roi(self, frame):
        """
        use simple blob to define ROI of an image - regions that potentially contain balls
        :param frame:
        :return: an image object with ROI
        """

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
        img_with_interesting_area = cv2.bitwise_and(img_with_interesting_area, img_with_interesting_area,
                                                    mask=mask_table)
        # cv2.imshow("img_with_interesting_area (filter out table background)", img_with_interesting_area)
        # cv2.waitKey(0)
        return img_with_interesting_area

    def detect_one_ball_from_img_with_roi(self, ball_id, img_with_roi):
        """
        use color range to find ball
        :param ball_id:
        :param img_with_roi:
        :return:
        """

        # print("ball_id", ball_id)
        img_with_roi_hsv = cv2.cvtColor(img_with_roi, cv2.COLOR_BGR2HSV)

        """
        apply ball color range on interesting area
        """
        ball_color_lower = np.array(self.balls[ball_id].hsv_color_lower, dtype="uint8")
        ball_color_upper = np.array(self.balls[ball_id].hsv_color_upper, dtype="uint8")
        mask_range = cv2.inRange(img_with_roi_hsv, ball_color_lower, ball_color_upper)
        mask_range = cv2.erode(mask_range, None, iterations=2)
        mask_range = cv2.dilate(mask_range, None, iterations=2)

        # # only display specific ball
        # image_with_ball = cv2.bitwise_and(img_with_roi, img_with_roi,
        #                                 mask=mask_range)
        # cv2.imshow(ball_id, image_with_ball)
        # cv2.waitKey(0)

        """
        find cnts, then ball center
        """
        cnts = cv2.findContours(mask_range.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(cnts) <= 0:
            return None

        # print("ball", ball_id, "found!")
        while len(cnts)>0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)

            ball_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            one_ball_rec = self.tmp_ball_tracking_rec_for_trajectory[ball_id]
            if len(one_ball_rec) > 0:
                prev_ball_center = one_ball_rec[0]
                if imutils.get_distance_of_two_points(ball_center, prev_ball_center) > 200:
                    cnts.remove(c)
                    continue

            return ball_center

        return None

    def real_time_tracking(self):
        camera = cv2.VideoCapture(self.video_file)
        fps = camera.get(cv2.CAP_PROP_FPS)
        print("Frames per second using camera.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
        frame_count = 0

        while True:
            frame_count += 1

            # grab the current frame
            (grabbed, frame) = camera.read()

            # if we are viewing a video and we did not grab a frame,
            # then we have reached the end of the video
            if self.video_file and not grabbed:
                break

            if self.resize:
                # resize the frame, blur it, and convert it to the HSV color space
                frame = imutils.resize(frame, width=800)
            # frame = cv2.GaussianBlur(frame, (11, 11), 0)

            """
            detect ROI
            """
            img_with_roi = self.get_img_with_roi(frame)
            # cv2.imshow("img_with_interesting_area (filter out table background)", img_with_interesting_area)
            # cv2.waitKey(0)

            """
            detect each ball using ball color range
            """
            for ball_id in self.balls:
                ball_center = self.detect_one_ball_from_img_with_roi(ball_id, img_with_roi)

                if ball_center:
                    self.tmp_ball_tracking_rec_for_trajectory[ball_id].appendleft(tuple(ball_center))
                    self.ball_tracking_rec_complete[ball_id].append(tuple(ball_center))
                else:
                    print("ball {} not found on frame {}!".format(ball_id, frame_count))
                    self.ball_tracking_rec_complete[ball_id].append(None)

                    # not_found_file_name = ball_id+"_not_found_frame_"+str(frame_count)+".png"
                    # cv2.imwrite(not_found_file_name, frame)

                """
                draw balls and trajectory
                """
                # cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 0), 2)
                cv2.circle(frame, ball_center, config.video_ball_radius, (0, 0, 255), 2)

                # trajectory
                tmp_ball_record = self.tmp_ball_tracking_rec_for_trajectory[ball_id]
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
            # print(self.ball_tracking_rec_for_real_time)

            key = cv2.waitKey(1) & 0xFF
            # if the 'q' key is pressed, stop the loop
            if key == ord("q"):
                cv2.destroyAllWindows()
                break

        camera.release()

    def draw_simple_trajectory(self):
        img = self.first_frame

        for ball_id in self.ball_tracking_rec_complete:
            one_ball_rec = self.ball_tracking_rec_complete[ball_id]
            one_ball_rec = [x for x in one_ball_rec if x is not None]

            for i in range(len(one_ball_rec) - 1):
                # print(one_ball_rec[i], one_ball_rec[i+1])

                cv2.circle(img, one_ball_rec[i], 3, self.balls[ball_id].trajectory_color, -1)
                cv2.line(img, one_ball_rec[i], one_ball_rec[i+1], self.balls[ball_id].trajectory_color, 1)

        # print("hello")
        cv2.imshow("draw_simple_trajectory", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_mean_shift(self):
        camera = cv2.VideoCapture(self.video_file)

        fps = camera.get(cv2.CAP_PROP_FPS)
        print ("Frames per second using camera.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

        frame_count = 0

        # first frame to determine the original ball (ball 0) and it's tracking window
        (grabbed, frame) = camera.read()
        if self.resize:
            frame = imutils.resize(frame, width=800)

        img_with_roi = self.get_img_with_roi(frame)
        ball_center = self.detect_one_ball_from_img_with_roi("0",img_with_roi)
        print(ball_center)
        c, r, w, h = ball_center[0]-5, ball_center[1]-5, 10, 10  # rectangle of ball area
        track_window = (c, r, w, h)

        cv2.rectangle(frame, (c, r), (c + w, r + h), 255, 2)
        cv2.imshow('test', frame)
        cv2.waitKey(0)

        # Create mask and normalized histogram
        roi = frame[r:r + h, c:c + w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # ball_color_lower = np.array(self.balls["0"].hsv_color_lower, dtype="uint8")
        # ball_color_upper = np.array(self.balls["0"].hsv_color_upper, dtype="uint8")
        # mask = cv2.inRange(hsv_roi, ball_color_lower, ball_color_upper)

        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1)

        # process each frame
        while True:
            ret, frame = camera.read()
            if self.resize:
                frame = imutils.resize(frame, width=800)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)
            x, y, w, h = track_window
            cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)

            cv2.imshow('Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # video_table = VideoTable("test_data/check0.png")
    # print(video_table.field_corners)

    video_file = "test_data/game1/video/1.mp4"

    myvideo1 = Video(video_file)
    # myvideo1.real_time_tracking()
    myvideo1.test_mean_shift()




