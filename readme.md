python3 colorAnalyze.py -m hsv -i test_data/game1/balls/ball2.png

mask_range = cv2.erode(mask_range, None, iterations=1)
mask_range = cv2.dilate(mask_range, None, iterations=1)
iteration need to be adjusteds