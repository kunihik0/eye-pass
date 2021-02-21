import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append("../")
sys.path.append("../tools/")
sys.path.append("../output_data/")

import numpy as np
import pandas as pd

from tools.evaluator import Evaluator

evaluator=Evaluator()

false_x_df = pd.read_csv("../output_data/chaos/x2.csv")
false_x_np = false_x_df.iloc[:,1:4].values
false_x_np = evaluator.get_derivative(false_x_np)

n_ans_win=0
n_false_win=0
false_list=[]
diff_list=[]
for i in range(3,8):
    
    ans_df = pd.read_csv("../output_data/eight/x"+str(i)+".csv")
    for j in range(3,8):
        if i==j:
            continue
        x_df = pd.read_csv("../output_data/eight/x"+str(j)+".csv")
        ans_np = ans_df.iloc[:,1:4].values
        x_np = x_df.iloc[:,1:4].values
       
        ans_np = evaluator.get_derivative(ans_np)
        x_np = evaluator.get_derivative(x_np)

        ansx_x = evaluator.calc_dtw(ans_np, x_np, evaluator.l2norm)
        ansx_false =evaluator.calc_dtw(ans_np, false_x_np, evaluator.l2norm)

        print("=========================")
        print(f"x{i}:x{j}:{ansx_x[-1][-1][0]}") 
        print(f"x{i}:false{ansx_false[-1][-1][0]}")
        if ansx_x[-1][-1][0] < ansx_false[-1][-1][0]:
            print("ans_x")
            if i < j:
                n_ans_win+=1
                diff=ansx_false[-1][-1][0]-ansx_x[-1][-1][0]
                diff_list.append(diff)
        else:
            print("false")
            if i < j:
                n_false_win+=1
                false_list.append([i,j])
        print("=========================")


print(f"n_ans_win {n_ans_win}")
print(f"n_false_win {n_false_win}")
print(false_list)
print(f"diff average {np.average(diff_list)}")
print(f"diff std {np.std(diff_list)}")
