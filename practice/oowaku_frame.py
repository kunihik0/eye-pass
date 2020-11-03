# https://www.youtube.com/watch?v=-VVih_oJ3jc&list=PL6Yc5OUgcoTlvHb5OfFLUJ90ofBuoU5g8&index=4
# part4

"""
discription
display "blinking" text when I blink.
judge eye's direction ,left or right or center

"""


from math import hypot
import dlib
import numpy as np
import cv2


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
    # hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    # ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    # length
    hor_line_length = hypot(
        (left_point[0]-right_point[0]), (left_point[1]-right_point[1]))
    ver_line_length = hypot(
        (center_top[0]-center_bottom[0]), (center_top[1]-center_bottom[1]))

    ratio = hor_line_length/ver_line_length

    return ratio


def get_gaze_ratio(eye_points, facial_landmarks):
    # Gaze detection
    left_eye_resion = np.array([(facial_landmarks.part(eye_points[0]).x,
                                 facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x,
                                 facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x,
                                 facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x,
                                 facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x,
                                 facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x,
                                 facial_landmarks.part(eye_points[5]).y)],
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


cap = cv2.VideoCapture(1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "../items/shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_PLAIN

left_eye_points = [36, 37, 38, 39, 40, 41]
right_eye_points = [42, 43, 44, 45, 46, 47]


default_side_ratio = [1.0, 2.4]
right_gaze_average_ratio = default_side_ratio[0]
left_gaze_average_ratio = default_side_ratio[1]

default_vertical_ratio = [1.3, 3.0]
lower_gaze_average_ratio = default_vertical_ratio[0]
upper_gaze_avarage_ratio = default_vertical_ratio[1]


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

        # Detect Blinking
        left_eye_ratio = get_blinking_ratio(left_eye_points, landmarks)
        right_eye_ratio = get_blinking_ratio(right_eye_points, landmarks)
        blinking_ratio = (left_eye_ratio+right_eye_ratio)/2

        if blinking_ratio > 6:
            cv2.putText(frame, "Blinking", (50, 150), font, 7, (255, 0, 0))

        # Gaze Detection
        gaze_side_ratio_left_eye, gaze_vertical_ratio_left_eye = \
            get_gaze_ratio(left_eye_points, landmarks)
        gaze_side_ratio_right_eye, gaze_vertical_ratio_right_eye = \
            get_gaze_ratio(right_eye_points, landmarks)

        gaze_side_ratio = (
            gaze_side_ratio_left_eye + gaze_side_ratio_right_eye)/2
        gaze_vertical_ratio = (
            gaze_vertical_ratio_left_eye+gaze_vertical_ratio_right_eye)/2

        # 画面右側：横の動き判定
        if gaze_side_ratio <= right_gaze_average_ratio:
            cv2.putText(frame, "right",
                        (50, 100), font, 2, (0, 0, 255), 3)
            new_frame[:] = (0, 0, 255)
        elif right_gaze_average_ratio < gaze_side_ratio < left_gaze_average_ratio:
            cv2.putText(frame, "center",
                        (50, 100), font, 2, (0, 0, 255), 3)
            new_frame[:] = (255, 0, 0)
        else:
            cv2.putText(frame, "left",
                        (50, 100), font, 2, (0, 0, 255), 3)
        cv2.putText(frame, str(gaze_side_ratio),
                    (50, 150), font, 2, (0, 0, 255), 3)

        # 画面左側：縦の動き判定
        if gaze_vertical_ratio <= lower_gaze_average_ratio:
            cv2.putText(frame, "lower",
                        (950, 100), font, 2, (0, 0, 255), 3)
            new_frame[:] = (0, 0, 255)
        elif lower_gaze_average_ratio < gaze_vertical_ratio < upper_gaze_avarage_ratio:
            cv2.putText(frame, "center",
                        (950, 100), font, 2, (0, 0, 255), 3)
            new_frame[:] = (255, 0, 0)
        else:
            cv2.putText(frame, "upper",
                        (950, 100), font, 2, (0, 0, 255), 3)
        cv2.putText(frame, str(gaze_vertical_ratio),
                    (950, 150), font, 2, (0, 0, 255), 3)

    cv2.imshow("Frame", frame)
    cv2.imshow("new_frame", new_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()
