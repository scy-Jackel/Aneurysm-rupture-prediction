import os
import xlrd
import xlwt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from config import config as cfg

np.set_printoptions(suppress=True)

def test_single(relation, id):
    if str(int(id)) not in relation.keys():
        return [], [], 0
    path = relation[str(int(id))]
    workbook = xlrd.open_workbook(path)
    sheet = workbook.sheets()[0]
    nrow = sheet.nrows
    ncol = sheet.ncols
    print(nrow, ncol)
    data=[]
    for i in range(1, nrow):
        for col_val in sheet.row_values(i):
            if 'NA' in str(col_val) or 'Not Loaded' in str(col_val):
                return [], [], 0
        data.append(sheet.row_values(i)[0:ncol])
    data_np = np.asarray(data , dtype=float)
    # res = np.mean(data_np, axis=0)
    choosen = data_np[:, [0,7,8,9,10,11,15,16,17,21,22]]
    # choosen = preprocessing.scale(choosen)
    mean_list = np.mean(choosen, axis=0)
    std_list = np.std(choosen, axis=0, ddof=1)
    return mean_list, std_list, nrow

def prepare():
    workbook = xlrd.open_workbook(cfg['totalExcel'])
    sheet = workbook.sheets()[0]
    nrow = sheet.nrows
    ncol = sheet.ncols
    ids = [] # save all id
    all_path = [] # save all path
    for i in range(1, nrow):
        id = sheet.row_values(i)[1]
        ids.append(int(id))
    single_paths = [os.path.join(cfg['singleDir'], f) for f in os.listdir(cfg['singleDir'])]
    multi_paths = [os.path.join(cfg['multiDir'], f) for f in os.listdir(cfg['multiDir'])]
    all_path = single_paths+multi_paths
    relation={}
    for id in ids:
        for path in all_path:
            if str(int(id)) in path:
               relation[str(id)]=path
    my_df = pd.DataFrame.from_dict(relation, orient='index')
    my_df.to_csv("./relations.csv", index=True, header=0) 
    print(my_df.shape)
    return relation


def fuse_data():
    df = pd.read_csv('./relations.csv', header=None)
    dct = df.to_dict(orient='list')
    # TODO
    pass

if __name__ == "__main__":
    # preprocess()
    fuse_data()
