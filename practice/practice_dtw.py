import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append("../")
sys.path.append("../tools/")
sys.path.append("../output_data/")

import numpy as np
import pandas as pd

from tools.evaluator import Evaluator

dir_name="circle"
false_x_df = pd.read_csv("../output_data/chaos/x1.csv")
false_x_np = false_x_df.iloc[:,1:4].values

n_ans_win=0
n_false_win=0
false_list=[]
diff_list=[]
for i in range(1,11):
    ans_df = pd.read_csv("../sample/"+dir_name+"/x"+str(i)+".csv")
    ans_np = ans_df.iloc[:,1:4].values
    for j in range(1,11):
        if i==j:
            continue
        x_df = pd.read_csv("../sample/"+dir_name+"/x"+str(j)+".csv")
        x_np = x_df.iloc[:,1:4].values
        evaluator = Evaluator()

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
