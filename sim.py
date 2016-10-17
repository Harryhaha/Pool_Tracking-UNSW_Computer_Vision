import cv2
import numpy as np

import config


class SimTable:
    def __init__(self):
        self.table_img_path = "test_data/pool_topview/topview1.gif"
        self.table_corners = np.array([[42, 611], [42, 36], [975, 36], [975, 611]], dtype="float32")


class SimBall:
    def __init__(self, ball_id, data=None):
        self.ball_id = ball_id
        self.ball_color = None
        self.ball_img = None

        if data:
            self.ball_color = data["ball_color"]
            self.ball_img = data["ball_image"]


class Sim:
    def __init__(self, tracking_ball_dic=None):
        """
        :param tracking_ball_dic: # {"0":[(111,222),(111,333),None,...], "3":..., ...}
        """

        self.sim_table = None
        self.sim_balls = {}
        self.tracking_ball_dic = tracking_ball_dic

        self.init_table()
        self.init_balls()

    def init_table(self):
        self.sim_table = SimTable()

    def init_balls(self):
        for ball_id in config.sim_ball_data:
            self.sim_balls[ball_id] = SimBall(ball_id, data=config.sim_ball_data[ball_id])

    def set_tracking_ball_dic(self, tracking_ball_dic):
        self.tracking_ball_dic = tracking_ball_dic

    def draw_simple_trajectory(self):
        img = cv2.imread(self.sim_table.table_img_path)

        for ball_id in self.tracking_ball_dic:
            one_ball_rec = self.tracking_ball_dic[ball_id]
            one_ball_rec = [x for x in one_ball_rec if x is not None]

            for i in range(len(one_ball_rec) - 1):
                # print(one_ball_rec[i], one_ball_rec[i+1])

                cv2.circle(img, one_ball_rec[i], 3, self.sim_balls[ball_id].ball_color, -1)
                cv2.line(img, one_ball_rec[i], one_ball_rec[i+1], self.sim_balls[ball_id].ball_color, 1)

        # print("hello")
        cv2.imshow("draw_simple_trajectory", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



