# https://www.youtube.com/watch?v=-VVih_oJ3jc&list=PL6Yc5OUgcoTlvHb5OfFLUJ90ofBuoU5g8&index=4
# part4

"""
discription
display "blinking" text when I blink.
judge eye's direction ,left or right or center

"""

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
from tools.indicator import Indicator


args = sys.argv
csv_file_name = "test.csv"
if len(args) > 1:
    csv_file_name = args[1] + ".csv"

cap = cv2.VideoCapture(1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "../items/shape_predictor_68_face_landmarks.dat")

default_side_ratio = [1.0, 2.4]

default_vertical_ratio = [1.3, 3.0]

start_time = time.time() 
data_list=[]
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

        indicator=Indicator(frame,gray,landmarks)
        blinking_ratio=indicator.blinking_indicator()

        gaze_side_ratio, gaze_vertical_ratio=indicator.gaze_indicator(frame,gray,new_frame,default_side_ratio)


        elapsed_time=time.time()-start_time
        # csv fileに保存
        data_list.append([elapsed_time, gaze_side_ratio, gaze_vertical_ratio, blinking_ratio])

    cv2.imshow("Frame", frame)
    cv2.imshow("new_frame", new_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
print(data_list)
csv_file_path = "../output_data/" + csv_file_name
header = ["time","gaze_side_ratio", "gaze_vertical_ratio", "blinking_ratio"]
data2csv(data_list=data_list,
         csv_file_path=csv_file_path, header=header)


cap.release()
cv2.destroyAllWindows()
