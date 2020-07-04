from math import hypot
import dlib
import numpy as np
import cv2

from .tool_from_youtube import *


cap = cv2.VideoCapture(1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "../items/shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_PLAIN

left_eye_points = [36, 37, 38, 39, 40, 41]
right_eye_points = [42, 43, 44, 45, 46, 47]


count = 0
sum_gaze_ratio = 0
while True:  # (10*30)*3=120 10:移行時間
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)

        gaze_ratio_left_eye = get_gaze_ratio(left_eye_points, landmarks)
        gaze_ratio_right_eye = get_gaze_ratio(right_eye_points, landmarks)
        gaze_ratio = (gaze_ratio_left_eye+gaze_ratio_right_eye)/2
        if count >= 30:
            sum_gaze_ratio += gaze_ratio
            count += 1

        cv2.putText(frame, str(gaze_ratio),
                    (50, 150), font, 2, (0, 0, 255), 3)

        count += 1
        if count >= 120:
            break

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    gaze_average_ratio = sum_gaze_ratio/90

print(gaze_average_ratio)
