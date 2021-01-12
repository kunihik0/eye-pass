import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append("../")
sys.path.append("../tools/")
sys.path.append("../output_data/")

import numpy as np
import pandas as pd

from tools.evaluator import Evaluator

dir_name="eight"
false_dir_name="square"

eight_list=[3,4,5,6,7,8]
circle_list=[3,8,4,1,2,5]
square_list=[1,2,3,4,5,6]
small_eight_list=[3,4,7,5,6,9]
small_square_list=[8,9,5,10,2,7]
small_circle_list=[1,2,3,4,5,6]
x_dict={"eight":eight_list,"circle":circle_list,"square":square_list,"small_eight":small_eight_list,"small_square":small_square_list,"small_circle":small_circle_list}

n_ans_win=0
n_false_win=0
false_list=[]
diff_list=[]

x_list=x_dict[dir_name]
ans_index=x_list.pop(0)    
false_x_df = pd.read_csv("../output_data/"+false_dir_name+"/x"+str(ans_index)+".csv")
false_x_np = false_x_df.iloc[:,1:4].values

dtw_ansx_x_list=[]
dtw_ansx_false_list=[]
dtw_x_false_list=[]
ansx_false=0
ans_df = pd.read_csv("../output_data/"+dir_name+"/x"+str(ans_index)+".csv")
for i in x_list:
    x_df = pd.read_csv("../output_data/"+dir_name+"/x"+str(i)+".csv")
    ans_np = ans_df.iloc[:,1:4].values
    x_np = x_df.iloc[:,1:4].values
    evaluator = Evaluator()

    ansx_x = evaluator.calc_dtw(ans_np, x_np, evaluator.l2norm)
    ansx_false =evaluator.calc_dtw(ans_np, false_x_np, evaluator.l2norm)
    x_false=evaluator.calc_dtw(x_np,false_x_np, evaluator.l2norm)
    dtw_ansx_x_list.append(ansx_x[-1][-1][0])
    dtw_ansx_false_list.append(ansx_false[-1][-1][0])
    dtw_x_false_list.append(x_false[-1][-1][0])
    print("=========================")
    print(f"x{i}:x{ans_index}:{ansx_x[-1][-1][0]}") 
    print(f"x{i}:false{ansx_false[-1][-1][0]}")
    if ansx_x[-1][-1][0] < ansx_false[-1][-1][0]:
        print("ans_x")
        n_ans_win+=1
        diff=ansx_false[-1][-1][0]-ansx_x[-1][-1][0]
        diff_list.append(diff)
    else:
        print("false")
        n_false_win+=1
        false_list.append([i,ans_index])
    print("=========================")


print(f"n_ans_win {n_ans_win}")
print(f"n_false_win {n_false_win}")
print(f"{100*n_ans_win/5}%")
print(false_list)
print(f"diff average {np.average(diff_list)}")
print(f"diff std {np.std(diff_list)}")

print("ans_x")
for i in dtw_ansx_x_list:
  print(i)

print("false_x")
print(dtw_ansx_false_list[0])
for i in dtw_x_false_list:
  print(i)
