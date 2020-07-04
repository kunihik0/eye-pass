# https://qiita.com/a2kiti/items/5ef16e8ff26f89044b7c
import cv2
import datetime


import cv2
import numpy as np
import dlib
import websocket


url = 'wss://9f44ff45115a.ngrok.io'
ws = websocket.create_connection(url)
cap = cv2.VideoCapture(1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "../items/shape_predictor_68_face_landmarks.dat")


while True:
    _, frame = cap.read()
    _, encimg = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    img_str = encimg.tostring()

    # 送受信
    ws.send_binary(img_str)
    body = ws.recv()

    # 文字列を画像にデコード
    data_np = np.frombuffer(body.encode(), dtype='uint8')
    decimg = cv2.imdecode(data_np, 3)
    frame = decimg

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()


"""
from google.colab import drive 
drive.mount('/content/drive')
################################
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

#############################
get_ipython().system_raw('./ngrok http 6006 &')
! curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

from math import hypot
import numpy as np
import cv2
import json
import bottle
import gevent
import dlib
from bottle.ext.websocket import GeventWebSocketServer
from bottle.ext.websocket import websocket


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "drive/My Drive/卒研/shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_PLAIN

left_eye_points = [36, 37, 38, 39, 40, 41]
right_eye_points = [42, 43, 44, 45, 46, 47]

app = bottle.Bottle()
@app.route('/', apply=[websocket])
def wsbin(ws):
    while True:
        body = ws.receive()
        if not body:
            break

        #文字列を画像にデコード
        data_np = np.frombuffer(body, dtype='uint8')
        decimg = cv2.imdecode(data_np, 3)

        #############何かの処理###############
        # out_img = decimg
        frame=decimg
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            # x, y = face.left(), face.top()
            # x1, y1 = face.right(), face.bottom()
            # cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

            landmarks = predictor(gray, face)

            # Detect Blinking
            left_eye_ratio = get_blinking_ratio(left_eye_points, landmarks)
            right_eye_ratio = get_blinking_ratio(right_eye_points, landmarks)
            blinking_ratio = (left_eye_ratio+right_eye_ratio)/2

            if blinking_ratio > 6:
                cv2.putText(frame, "Blinking", (50, 150), font, 7, (255, 0, 0))

            # Gaze detection
            left_eye_resion = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                        (landmarks.part(37).x, landmarks.part(37).y),
                                        (landmarks.part(38).x, landmarks.part(38).y),
                                        (landmarks.part(39).x, landmarks.part(39).y),
                                        (landmarks.part(40).x, landmarks.part(40).y),
                                        (landmarks.part(41).x, landmarks.part(41).y)],
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
            left_eye = cv2.bitwise_and(gray, gray, mask=mask)

            # 目を長方形windowで抜き出し
            min_x = np.min(left_eye_resion[:, 0])
            max_x = np.max(left_eye_resion[:, 0])
            min_y = np.min(left_eye_resion[:, 1])
            max_y = np.max(left_eye_resion[:, 1])
            gray_eye = left_eye[min_y: max_y, min_x:max_x]

            # 白黒にするための閾値を決めて（今回は70）、全体を白黒にする
            _, threshold_eye = cv2.threshold(
                gray_eye, 70, 255, cv2.THRESH_BINARY)

            threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
            eye = cv2.resize(gray_eye, None, fx=5, fy=5)
            out_img=eye

        #############何かの処理###############

        #文字列にエンコード
            _, encimg = cv2.imencode(".jpg", out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            img_str = encimg.tostring()
            ws.send(img_str)

app.run(host='0.0.0.0', port=6006, server=GeventWebSocketServer)
"""
