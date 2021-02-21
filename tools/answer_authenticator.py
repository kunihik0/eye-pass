import math
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append("../")
sys.path.append("../tools/")
sys.path.append("../answer_data/")

import numpy as np
import pandas as pd

from tools.evaluator import Evaluator



class Answer_Authenticator(object):
    def __init__(self):
        pass

    def authenticator(self, np1, np2, threshold):
        dtw = Evaluator().calc_dtw
        func_dist = Evaluator().l2norm

        distance = dtw(np1, np2, func_dist)[-1][-1][0]

        self.judgment(distance, threshold)

    def judgment(self, distance, threshold):
        if distance < threshold:
            print("succes!")
        else:
            print("failure")
