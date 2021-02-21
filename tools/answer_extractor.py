import copy
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
from tools._data2csv import data2csv, data2new_csv
args = sys.argv
answer_csv_path = "../answer_data/answer.csv"
if len(args) > 1:
    csv_file_name = "../answer_data/" + args[1] + ".csv"


def make_answer():
    dtw = Evaluator().calc_dtw
    func_dist = Evaluator().l2norm

    data_list = ["x1","x2","x3","x4","x5","x6","x7","x8","x9","x10"] 
    answer_data = None
    distance_list = [[0]*10 for _ in range(10)]

    #10個のx~.csvデータを読み込み。
    df_dict = {}
    for i in data_list:
        csv_file_path = "../answer_data/" + i + ".csv"
        df = pd.read_csv(csv_file_path)
        df_dict[i] = df

    #総当たりで距離計算してdistance_listに格納
    for i in range(9):
        data1 = data_list[i]
        df1 = df_dict[data1]
        df1_np = df1.iloc[:,1:4].values
        for j in range(i+1, 10):
            data2 = data_list[j]
            df2 = df_dict[data2]
            df2_np = df2.iloc[:,1:4].values

            if i == j:
                continue

            distance = dtw(df1_np, df2_np, func_dist)
            distance_list[i][j] = distance[-1][-1][0]
            distance_list[j][i] = distance[-1][-1][0]

    #各データの距離計算した結果のばらつき具合を求める
    var_list = [0]*10
    tmp_distance_list = copy.copy(distance_list)
    for i, data in enumerate(tmp_distance_list):
        data.pop(i)
        var = np.var(data)
        var_list[i] = var

    #ばらつきの一番小さかったデータをanswerデータとしそのindexを返す
    min_idx = 0
    min_var = var_list[0]
    for i in range(1,10):
        compared_var = var_list[i]
        if min_var > compared_var:
            min_var = compared_var
            min_idx = i

    #answerデータとそれ以外のデータの距離の平均、標準偏差を出す。
    #欠損値が混ざってる可能性があるため距離の小さい上位5個取り出す。
    #[:5]ではなく[1:6]なのは最小値が0だから。
    min_distance_data = distance_list[min_idx]
    min_distance_data = sorted(min_distance_data)[1:6]
    min_average = np.average(min_distance_data)
    min_std = np.std(min_distance_data)

    min_data = data_list[min_idx]
    min_df = df_dict[min_data]

    columns = min_df.columns.values
    df_values = min_df.values.tolist()

    answer_list=[[min_average, min_std]]
    answer_list.append(columns)
    answer_list += df_values

    
    data2new_csv(answer_list, answer_csv_path)

