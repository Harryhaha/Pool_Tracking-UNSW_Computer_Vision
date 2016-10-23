video_table_data = {
    "hsv_color_lower": (95, 100, 0),
    "hsv_color_upper": (105, 255, 255)
}

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
        {"hsv_color_lower": (99, 0, 50),
         "hsv_color_upper": (119, 255, 150),
         "trajectory_color": (112, 77, 49)
         }
}

sim_ball_data = {
    "0":
        {"ball_color": (255, 255, 255), "ball_image": None},
    "3":
        {"ball_color": (134, 0, 252), "ball_image": None},
    "2":
        {"ball_color": (112, 77, 49), "ball_image": None}
}

first_frame_save_path = "test_data/first_frame.png"

video_ball_radius = 10
sim_ball_radius = 10

max_move_dis = 100

black_v_max = 10

white_s_max = 10

mean_hue_variance = 10

balls_avg_BGR = {
    "0": (255, 255, 255),  # This is used to match write ball or stripped balls
    "1": (101.3061224489796, 182.0, 240.3673469387755),
    "2": (112.15416666666667, 77.475, 49.62083333333333),
    "3": (56.38392857142857, 77.91964285714286, 201.66071428571428),
    "5": (131.54395604395606, 149.87912087912088, 229.03846153846155)
}