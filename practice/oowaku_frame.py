# https://www.youtube.com/watch?v=-VVih_oJ3jc&list=PL6Yc5OUgcoTlvHb5OfFLUJ90ofBuoU5g8&index=4
# part4

"""
discription
display "blinking" text when I blink.
judge eye's direction ,left or right or center

"""

from copy import copy
from math import hypot
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append("../")
sys.path.append("../tools/")

import cv2
import dlib
import numpy as np

from tools._data2csv import data2csv
from tools.tool_from_youtube import midpoint, get_blinking_ratio, get_gaze_ratio
from tools.detector import Detector
from tools.getter import Getter
from tools.discriminator import Discriminator
from tools.indicator import Indicator


args = sys.argv
csv_file_name = "test.csv"
if len(args) > 1:
    csv_file_name = args[1] + ".csv"

cap = cv2.VideoCapture(1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "../items/shape_predictor_68_face_landmarks.dat")

#font = cv2.FONT_HERSHEY_PLAIN

#discriminator = Discriminator(font)

#left_eye_points = [36, 37, 38, 39, 40, 41]
#right_eye_points = [42, 43, 44, 45, 46, 47]


default_side_ratio = [1.0, 2.4]
#right_gaze_average_ratio = default_side_ratio[0]
#left_gaze_average_ratio = default_side_ratio[1]

default_vertical_ratio = [1.3, 3.0]
#lower_gaze_average_ratio = default_vertical_ratio[0]
#upper_gaze_avarage_ratio = default_vertical_ratio[1]

start_time = time.time() 

while True:
    _, frame = cap.read()
    new_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)

        #detector_tool = Detector(landmarks,left_eye_points,right_eye_points)
        #blinking_ratio = detector_tool.blinking_detector()
        # Detect Blinking
        #left_eye_ratio = get_blinking_ratio(left_eye_points, landmarks)
        #right_eye_ratio = get_blinking_ratio(right_eye_points, landmarks)
        #blinking_ratio = (left_eye_ratio+right_eye_ratio)/2

        #if blinking_ratio > 6:
        #    cv2.putText(frame, "Blinking", (50, 150), font, 7, (255, 0, 0))

        indicator=Indicator(frame,gray,landmarks)
        blinking_ratio=indicator.blinking_indicator()

        gaze_side_ratio, gaze_vertical_ratio=indicator.gaze_indicator(frame,gray,new_frame,default_side_ratio)

        # Gaze Detection
        #gaze_side_ratio_left_eye, gaze_vertical_ratio_left_eye = \
        #    get_gaze_ratio(frame, gray, left_eye_points, landmarks)
        #gaze_side_ratio_right_eye, gaze_vertical_ratio_right_eye = \
        #    get_gaze_ratio(frame, gray, right_eye_points, landmarks)

        #gaze_side_ratio = (
        #    gaze_side_ratio_left_eye + gaze_side_ratio_right_eye)/2
        #gaze_vertical_ratio = (
        #    gaze_vertical_ratio_left_eye+gaze_vertical_ratio_right_eye)/2

#        # 画面右側：横の動き判定
#        if gaze_side_ratio <= right_gaze_average_ratio:
#            cv2.putText(frame, "right",
#                        (50, 100), font, 2, (0, 0, 255), 3)
#            new_frame[:] = (0, 0, 255)
#        elif right_gaze_average_ratio < gaze_side_ratio < left_gaze_average_ratio:
#            cv2.putText(frame, "center",
#                        (50, 100), font, 2, (0, 0, 255), 3)
#            new_frame[:] = (255, 0, 0)
#        else:
#            cv2.putText(frame, "left",
#                        (50, 100), font, 2, (0, 0, 255), 3)
#        cv2.putText(frame, str(gaze_side_ratio),
#                    (50, 150), font, 2, (0, 0, 255), 3)
#
#        # 画面左側：縦の動き判定
#        if gaze_vertical_ratio <= lower_gaze_average_ratio:
#            cv2.putText(frame, "lower",
#                        (950, 100), font, 2, (0, 0, 255), 3)
#            new_frame[:] = (0, 0, 255)
#        elif lower_gaze_average_ratio < gaze_vertical_ratio < upper_gaze_avarage_ratio:
#            cv2.putText(frame, "center",
#                        (950, 100), font, 2, (0, 0, 255), 3)
#            new_frame[:] = (255, 0, 0)
#        else:
#            cv2.putText(frame, "upper",
#                        (950, 100), font, 2, (0, 0, 255), 3)
#        cv2.putText(frame, str(gaze_vertical_ratio),
#                    (950, 150), font, 2, (0, 0, 255), 3)

        elapsed_time=time.time()-start_time
        # csv fileに保存
        data_list = [elapsed_time, gaze_side_ratio, gaze_vertical_ratio, blinking_ratio]
        csv_file_path = "../output_data/" + csv_file_name
        header = ["time","gaze_side_ratio", "gaze_vertical_ratio", "blinking_ratio"]
        data2csv(data_list=data_list,
                 csv_file_path=csv_file_path, header=header)

    cv2.imshow("Frame", frame)
    cv2.imshow("new_frame", new_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
