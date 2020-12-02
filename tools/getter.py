from math import hypot
import dlib
import numpy as np
import cv2


def get_midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)


def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(
        eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(
        eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)

    # point
    center_top = get_midpoint(facial_landmarks.part(
        eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = get_midpoint(facial_landmarks.part(
        eye_points[5]), facial_landmarks.part(eye_points[4]))

    # line
    # hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    # ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    # length
    hor_line_length = hypot(
        (left_point[0]-right_point[0]), (left_point[1]-right_point[1]))
    ver_line_length = hypot(
        (center_top[0]-center_bottom[0]), (center_top[1]-center_bottom[1]))

    ratio = hor_line_length/ver_line_length

    return ratio


def get_gaze_ratio(frame, gray, eye_points, facial_landmarks):
    # Gaze detection
    left_eye_resion = np.array([(facial_landmarks.part(0).x, facial_landmarks.part(0).y),
                                (facial_landmarks.part(1).x,
                                 facial_landmarks.part(1).y),
                                (facial_landmarks.part(2).x,
                                 facial_landmarks.part(2).y),
                                (facial_landmarks.part(3).x,
                                 facial_landmarks.part(3).y),
                                (facial_landmarks.part(4).x,
                                 facial_landmarks.part(4).y),
                                (facial_landmarks.part(5).x, facial_landmarks.part(5).y)],
                               np.int32)
    # cv2.polylines(frame, [left_eye_resion], True, (0, 0, 255), 2)

    # カメラのdefault sizeの真っ黒window作成　
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)

    # 目を囲む
    cv2.polylines(mask, [left_eye_resion], True, 255, 2)
    # 目ん玉塗りつぶす
    cv2.fillPoly(mask, [left_eye_resion], 255)
    # gray(顔全体)を目の部分だけ抽出
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    # 目を長方形windowで抜き出し
    min_x = np.min(left_eye_resion[:, 0])
    max_x = np.max(left_eye_resion[:, 0])
    min_y = np.min(left_eye_resion[:, 1])
    max_y = np.max(left_eye_resion[:, 1])
    gray_eye = eye[min_y: max_y, min_x:max_x]

    # 白黒にするための閾値を決めて（今回は70）、全体を白黒にする
    _, threshold_eye = cv2.threshold(
        gray_eye, 80, 255, cv2.THRESH_BINARY)

    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0:height, 0:int(width/2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0:height, int(width/2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    try:
        gaze_side_ratio = left_side_white/right_side_white
    except ZeroDivisionError:
        gaze_side_ratio = left_side_white/(1e-4)

    upper_side_threshold = threshold_eye[int(height/2):height, 0:width]
    upper_side_white = cv2.countNonZero(upper_side_threshold)

    lower_side_threshold = threshold_eye[0:int(height/2), 0:width]
    lower_side_white = cv2.countNonZero(lower_side_threshold)

    try:
        gaze_vertical_ratio = upper_side_white/lower_side_white
    except ZeroDivisionError:
        gaze_vertical_ratio = upper_side_white/(1e-4)

    return gaze_side_ratio, gaze_vertical_ratio
