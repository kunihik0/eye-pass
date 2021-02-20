import os                
import sys               
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append("../")
sys.path.append("../tools/")     

import numpy as np
#from fastdtw

class Evaluator(object):
    def __init__(self):
        pass

    def first(self, x):
        return x[0]

    def second(self, x):
        return x[1]

    def l2norm(self, ansx, x):
        dist = np.linalg.norm(ansx - x)
        return dist

    def l2norm_2d(self, ansx, ansy, x, y):
        dist = ((ansx - x) ** 2 + (ansy - y) ** 2) ** 0.5
        return dist


    def l2norm_3d(self, ansx, ansy,ansz, x, y, z):
        dist=((ansx - x) ** 2 + (ansy - y) ** 2 + (ansz - z) ** 2) ** 0.5 
        return dist

    def min_val(self, v1, v2, v3):
        f1 = self.first(v1)
        f2 = self.first(v2)
        f3 = self.first(v3)

        if f1 <= min(f2, f3):
            return v1, 0
        elif f2 <= f3:
            return v2, 1
        else:
            return v3, 2

    def calc_dtw(self, ansx, x, dist_func):
        s = len(ansx)
        t = len(x)

        m = [[0 for j in range(t)] for i in range(s)]
        m[0][0] = (dist_func(ansx[0], x[0]), (-1, -1))

        for i in range(1, s):
            m[i][0] = (m[i-1][0][0] + dist_func(ansx[i], x[0]), (i-1,0))
        for j in range(1,t):
            m[0][j] = (m[0][j-1][0] + dist_func(ansx[0], x[j]), (0, j-1))

        for i in range(1, s):
            for j in range(1, t):
                minimum, index = self.min_val(m[i-1][j], m[i][j-1], m[i-1][j-1])
                indexes = [(i-1, j), (i, j-1), (i-1, j-1)]
                m[i][j] = (self.first(minimum) + dist_func(ansx[i], x[j]), indexes[index])
        
        return m 

    def get_derivative(self,t):
        l_t=len(t)
        new_list=[]
        for i in range(1,l_t-1):
            a=t[i]-t[i-1]
            b=(t[i+1]-t[i-1])/2
            new_list.append((a+b)/2)
        new_list[0]=new_list[1]
        new_list[-1]=new_list[-2]
        derivatived_array=np.array(new_list)

        return derivatived_array
