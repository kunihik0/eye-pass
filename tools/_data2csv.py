import csv
import pathlib
import os


def data2new_csv(data_list, csv_file_path="/tmp", header=[]):
    """
    読み取ったデータをcsvファイルに保存する。
    もし同じ名前のファイルがあっても前のファイルを消して
    新しいファイルを作る。

    Parameters
    ----------
    data_list : list of int
        csvファイルに記入する通りの列の並びで作れられたデータの集まり。
    csv_file_path : str
        csvファイルの保存先。
    header : list
        csvファイルに記入する列のヘッダー。
        何も入れない場合は0から埋めてく。

    """

    with open(csv_file_path, 'w') as f:
        writer = csv.writer(f)


        writer.writerows(data_list)


def data2csv(data_list, csv_file_path="/tmp", header=[]):
    """
    読み取ったデータをcsvファイルに保存する。
    もし同じ名前のファイルが存在した場合は、
    データを消さずに足していく。

    Parameters
    ----------
    data_list : list of int
        csvファイルに記入する通りの列の並びで作れられたデータの集まり。
    csv_file_path : str
        csvファイルの保存先。
    header : list 
        csvファイルに記入する列のヘッダー。
        何も入れない場合は0から埋めてく。

    """

    is_new_file = False
    if os.path.exists(csv_file_path) == False:
        is_new_file = True

    if is_new_file:
        with open(csv_file_path, 'w') as f:
            writer = csv.writer(f)

#            if len(header) > 0:
#                writer.writerow(header)
#            else:
#                header = [i for i in range(len(data_list[0]))]
#                writer.writerow(header)

            writer.writerows(data_list)

    else:
        with open(csv_file_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerows(data_list)

    print(is_new_file)


#data2csv(data_list=[0, 2],
#         csv_file_path="../output_data/test.csv")
