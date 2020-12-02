import os                
import sys               
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append("../")
sys.path.append("../tools/")                                   
                         
from tools.detector import Detector
from tools.discriminator import Discriminator


class Indicator(object):
    def __init__(self,frame,gray,landmarks):
        self.detector=Detector(frame,gray,landmarks)
        self.discriminator=Discriminator(frame,landmarks)

    def blinking_indicator(self):
        blinking_ratio=self.detector.blinking_detector()
        self.discriminator.blinking_discriminator(blinking_ratio, threshold=6)
        
        return blinking_ratio

    def gaze_indicator(self,frame,gray,new_frame,default_side_ratio):
        gaze_side_ratio, gaze_vertical_ratio=self.detector.gaze_detector()
        self.discriminator.side_discriminator(new_frame,default_side_ratio,gaze_side_ratio)
        self.discriminator.vertical_discriminator(new_frame,default_side_ratio,gaze_vertical_ratio)
        return gaze_side_ratio, gaze_vertical_ratio







