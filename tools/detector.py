import os
import sys
import time
 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append("../")
sys.path.append("../tools/")
 
import cv2
import dlib
import numpy as np

from tools.getter import get_midpoint, get_blinking_ratio, get_gaze_ratio

class Detector(object):
    def __init__(self,frame,gray,landmarks):
        self.frame=frame
        self.gray=gray
        self.landmarks=landmarks
        self.left_eye_points=[36, 37, 38, 39, 40, 41]
        self.right_eye_points= [42, 43, 44, 45, 46, 47]

    def blinking_detector(self):
        left_eye_ratio = get_blinking_ratio(self.left_eye_points, self.landmarks)
        right_eye_ratio = get_blinking_ratio(self.right_eye_points, self.landmarks)
        blinking_ratio = (left_eye_ratio+right_eye_ratio)/2

        return blinking_ratio

    def gaze_detector(self):
        # Gaze Detection
        gaze_side_ratio_left_eye, gaze_vertical_ratio_left_eye = \
            get_gaze_ratio(self.frame, self.gray, self.left_eye_points, self.landmarks)
        gaze_side_ratio_right_eye, gaze_vertical_ratio_right_eye = \
            get_gaze_ratio(self.frame, self.gray, self.right_eye_points, self.landmarks)
 
        gaze_side_ratio = (
            gaze_side_ratio_left_eye + gaze_side_ratio_right_eye)/2
        gaze_vertical_ratio = (
            gaze_vertical_ratio_left_eye+gaze_vertical_ratio_right_eye)/2

        return gaze_side_ratio, gaze_vertical_ratio

    def key_detector(self,key):
        pass
