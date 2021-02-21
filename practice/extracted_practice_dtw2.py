"""
同じ種類で総当たり
類似度比べるため
false_np使わない
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append("../")
sys.path.append("../tools/")
sys.path.append("../output_data/")

import numpy as np
import pandas as pd

from tools.evaluator import Evaluator

eight_list=[3,4,5,6,7,8]
circle_list=[3,8,4,1,2,5]
square_list=[1,2,3,4,5,6]
small_eight_list=[3,4,7,5,6,9]
small_square_list=[8,9,5,10,2,7]
small_circle_list=[1,2,3,4,5,6]
x_dict={"eight":eight_list,"circle":circle_list,"square":square_list,"small_eight":small_eight_list,"small_circle":small_circle_list,"small_square":small_square_list}

all_df=pd.DataFrame(x_dict.keys())
evaluator = Evaluator()
for dir_name in x_dict.keys():
    output_name=dir_name+"_dtw"
    csv_file_path="../items/"+output_name+".csv"
    x_list=x_dict[dir_name]
    all_list=[]
    for i in range(1,11):
        print(i)
        x1_df = pd.read_csv("../sample/"+dir_name+"/x"+str(i)+".csv")
        x1_np = x1_df.iloc[:,1:4].values
        tmp_list=[]
        for j in range(1,11):
            if i==j:
                print("###")
                tmp_list.append(None)
                continue

            x2_df=pd.read_csv("../sample/"+dir_name+"/x"+str(j)+".csv")
            x2_np=x2_df.iloc[:,1:4].values
            
            dtw_x1_x2 = evaluator.calc_dtw(x1_np, x2_np, evaluator.l2norm)    
            print(dtw_x1_x2[-1][-1][0])
            tmp_list.append(dtw_x1_x2[-1][-1][0])
        all_list.append(tmp_list)

    tmp_df=pd.DataFrame(all_list)
    all_df=pd.concat([all_df,tmp_df])
all_df.to_csv("../items/all2.csv")
"print(all_df.iloc[:,:])"
