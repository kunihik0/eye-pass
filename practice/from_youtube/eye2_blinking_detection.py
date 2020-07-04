# https://www.youtube.com/watch?v=mMObcjHs59E&list=PL6Yc5OUgcoTlvHb5OfFLUJ90ofBuoU5g8&index=2
# part2

"""
discription
show eyes's horizontal line and vetical line.
display "blinking" text when I close my eys. 

"""

import cv2
import numpy as np
import dlib
from math import hypot
cap = cv2.VideoCapture(1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "../../items/shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_PLAIN

left_eye_points = [36, 37, 38, 39, 40, 41]
right_eye_points = [42, 43, 44, 45, 46, 47]


def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(
        eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(
        eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)

    # point
    center_top = midpoint(facial_landmarks.part(
        eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(
        eye_points[5]), facial_landmarks.part(eye_points[4]))

    # line
    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    # length
    hor_line_length = hypot(
        (left_point[0]-right_point[0]), (left_point[1]-right_point[1]))
    ver_line_length = hypot(
        (center_top[0]-center_bottom[0]), (center_top[1]-center_bottom[1]))

    ratio = hor_line_length/ver_line_length

    return ratio


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        # x, y = face.left(), face.top()
        # x1, y1 = face.right(), face.bottom()
        # cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
        landmarks = predictor(gray, face)

        left_eye_ratio = get_blinking_ratio(left_eye_points, landmarks)
        right_eye_ratio = get_blinking_ratio(right_eye_points, landmarks)
        blinking_ratio = (left_eye_ratio+right_eye_ratio)/2

        if blinking_ratio > 6:
            cv2.putText(frame, "Blinking", (50, 150), font, 7, (255, 0, 0))

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
