import os
import sys
import time
 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append("../")
sys.path.append("../tools/")               
 
import cv2     
import dlib    
import numpy as np

from tools.detector import Detector

class Discriminator(object):
    def __init__(self, frame, landmarks):
        self.font=cv2.FONT_HERSHEY_PLAIN  
        self.frame=frame
        self.landmarks=landmarks

    def side_discriminator(self,new_frame,default_side_ratio,gaze_side_ratio):
        right_gaze_average_ratio = default_side_ratio[0]
        left_gaze_average_ratio = default_side_ratio[1]

        # 画面右側：横の動き判定
        if gaze_side_ratio <= right_gaze_average_ratio:
            cv2.putText(self.frame, "right",
                        (50, 100), self.font, 2, (0, 0, 255), 3)
            new_frame[:] = (0, 0, 255)
        elif right_gaze_average_ratio < gaze_side_ratio < left_gaze_average_ratio:
            cv2.putText(self.frame, "center",
                        (50, 100), self.font, 2, (0, 0, 255), 3)
            new_frame[:] = (255, 0, 0)
        else:
            cv2.putText(self.frame, "left",
                        (50, 100), self.font, 2, (0, 0, 255), 3)
        cv2.putText(self.frame, str(gaze_side_ratio),
                    (50, 150), self.font, 2, (0, 0, 255), 3)

    def vertical_discriminator(self,new_frame,default_vertical_ratio,gaze_vertical_ratio):
        lower_gaze_average_ratio = default_vertical_ratio[0]
        upper_gaze_avarage_ratio = default_vertical_ratio[1]

        # 画面左側：縦の動き判定
        if gaze_vertical_ratio <= lower_gaze_average_ratio:
            cv2.putText(self.frame, "lower",
                        (950, 100), self.font, 2, (0, 0, 255), 3)
            new_frame[:] = (0, 0, 255)
        elif lower_gaze_average_ratio < gaze_vertical_ratio < upper_gaze_avarage_ratio:
            cv2.putText(self.frame, "center",
                        (950, 100), self.font, 2, (0, 0, 255), 3)
            new_frame[:] = (255, 0, 0)
        else:
            cv2.putText(self.frame, "upper",
                        (950, 100), self.font, 2, (0, 0, 255), 3)
        cv2.putText(self.frame, str(gaze_vertical_ratio),
                    (950, 150), self.font, 2, (0, 0, 255), 3)

    def blinking_discriminator(self,blinking_ratio,threshold=6):
        if blinking_ratio > threshold:
            cv2.putText(self.frame, "Blinking", (50, 150), self.font, 7, (255, 0, 0)) 

