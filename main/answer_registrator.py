
from collections import deque
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append("../")
sys.path.append("../tools/")

import cv2
import dlib
import numpy as np

from tools._data2csv import data2csv, data2new_csv
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

directions_list = deque([
                  "x1","x2","x3","x4","x5","x6","x7","x8","x9","x10",
                  ])

start_time = time.time() 
data_list = []
data_dict = {}
type_count = 0
is_first = True
is_collecting_data = False
phase=None

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
        if is_collecting_data:
            data_list.append([elapsed_time, gaze_side_ratio, gaze_vertical_ratio, blinking_ratio])

    if is_first:
      is_first = False
      phase = directions_list.popleft()
      print("I'm ready. Please push Enter Button.")

    cv2.imshow("Frame", frame)
#    cv2.imshow("new_frame", new_frame)
    key = cv2.waitKey(1)
    if key > 0 and key!=27:
        
        if is_collecting_data:
            data_dict[phase] = data_list
            data_list = []
            is_collecting_data = False
            print("end", phase, "phase")
            print("")
            if directions_list:
                phase = directions_list.popleft()
                print(phase, "phase is ready")
            else:
                key=27

        else:
            is_collecting_data = True
            print("start", phase, "phase")        

    if key == 27:
        break

header = ["time","gaze_side_ratio", "gaze_vertical_ratio", "blinking_ratio"]
for key in data_dict.keys():
    csv_file_path = "../answer_data/" + key + ".csv"
    data2new_csv(data_list = data_dict[key],
             csv_file_path = csv_file_path, header=header)


cap.release()
cv2.destroyAllWindows()
