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
evaluator = Evaluator()
x_list=["eight","circle","square","small_eight","small_circle","small_square"]
threshold_dict={"eight":65.6,"circle":26.6,"square":46.6,"small_eight":43.7,"small_circle":16.4,"small_square":30.6}
all_list=[[0]*12 for _ in range(60)]
for dir_name in x_list:
    dn_idx=x_list.index(dir_name)
    threshold=threshold_dict[dir_name]
    for i in range(1,11):
        x_df=pd.read_csv("../sample/"+dir_name+"/x"+str(i)+".csv")
        x_np=x_df.iloc[:,1:4].values
        for dir_name2 in x_list:
            dn2_idx=x_list.index(dir_name2)
            for j in range(1,11):
                if i==j and dir_name==dir_name2:
                    continue
              
                y_df=pd.read_csv("../sample/"+dir_name2+"/x"+str(j)+".csv")
                y_np=y_df.iloc[:,1:4].values

                dtw_xy = evaluator.calc_dtw(x_np, y_np, evaluator.l2norm)
                ans = dtw_xy[-1][-1][0]

                if ans < threshold:
                    idx1=dn_idx*10+i-1
                    idx2=2*dn2_idx
                    all_list[idx1][idx2]+=1
                else:
                    idx1=dn_idx*10+i-1
                    idx2=2*dn2_idx+1
                    all_list[idx1][idx2]+=1

all_df=pd.DataFrame(all_list)
all_df.to_csv("../items/all1-2.csv")
print(all_df.iloc[:,:])
