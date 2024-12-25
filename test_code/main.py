import os
import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
import xlrd
import xlwt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn import svm

from config import config as cfg
from preprocess import prepare, test_single

def read_data():
    path = cfg['totalExcel']
    workbook = xlrd.open_workbook(path)
    sheet = workbook.sheets()[0]
    nrow = sheet.nrows
    ncol = sheet.ncols
    print(nrow, ncol)
    x = []
    y = []
    relation = prepare()
    for i in range(1, nrow):
        skip = False
        for j in range(0, ncol):
            if sheet.row_values(i)[j] == 'NA':
                skip = True
                break
        if skip:
            continue
        id = sheet.row_values(i)[1]
        mean_list, std_list, line_num = test_single(relation, id)
        if line_num==0:
            continue
        label = sheet.row_values(i)[2]
        props = sheet.row_values(i)[3: ncol]
        props = props+mean_list.tolist()+std_list.tolist()+[line_num]
        y.append(label)
        x.append(props)
    x_np = np.asarray(x, dtype=float)
    x_np = preprocessing.scale(x_np)
    x = [i for i in x_np]
    list1 = []
    list2 = []
    for i in range(len(x)):
        if y[i] == 1:
            list1.append(x[i])
        if y[i] == 2:
            list2.append(x[i])
    print("data  unify complete.")
    return list1, list2


def random_select(x, y,  total_size):
    t = time.time()
    np.random.seed(int(str(t % 1)[2:7]))
    sampled_x = random.sample(x, total_size)
    sampled_y = random.sample(y, total_size)
    return sampled_x, sampled_y


plot_cm = True


def classify(list1, list2, epoch):

    sample_size = min(len(list1), len(list2))
    # print("sample size:", sample_size)
    sample1, sample2 = random_select(list1, list2, sample_size)

    x = sample1+sample2
    y = [1 for i in range(len(sample1))]+[2 for i in range(len(sample2))]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, stratify=y)
    model = LinearSVC(max_iter=1e8) # 0.774
    # model = svm.SVC(C=1, kernel='poly', degree=10, gamma='auto',decision_function_shape='ovr')
    model.fit(x_train, y_train)
    # Calculate Test Prediction
    y_pred = model.predict(x_test)

    res = model.score(x_test, y_test)

    # Plot Confusion Matrix and ROC curve
    if plot_cm:
        plt.clf()
        classes = list(set(y_test))
        # print("classes: ", classes)
        confusion = confusion_matrix(y_test, y_pred)

        indices = range(len(confusion))
        plt.xticks(indices, classes)
        plt.yticks(indices, classes)
        plt.xlabel('predict')
        plt.ylabel('true')

        plt.imshow(confusion, cmap=plt.cm.Blues)
        plt.colorbar()
        for first_index in range(len(confusion)):
            for second_index in range(len(confusion[first_index])):
                plt.text(second_index, first_index,
                         confusion[first_index][second_index], va='center', ha='center')

        plt.savefig("./svmfigures/confusion"+str(epoch)+".jpg")
        plt.clf()
        # ROC
        y_test = np.asarray(y_test, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        for i in range(len(y_pred)):
            if y_pred[i] == 2:
                y_pred[i] = 0
        fpr, tpr, threshold = roc_curve(y_test, y_pred, pos_label=1)
        roc_auc = auc(fpr, tpr)
        print("epoch", epoch, "roc:", roc_auc)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('classification')
        plt.legend(loc="lower right")
        plt.savefig("./svmfigures/roc"+str(epoch)+".jpg")
        plt.close()

    return res


def save_data():
    list1, list2 = read_data()
    np1 = np.asarray(list1)
    np2 = np.asarray(list2)
    np.save("list1.npy", np1)
    np.save("list2.npy", np2)


def load_data():
    np1 = np.load("list1.npy")
    np2 = np.load("list2.npy")
    list1 = np1.tolist()
    list2 = np2.tolist()
    return list1,list2


if __name__ == "__main__":
    # save_data()
    list1, list2 = load_data()
    print("list1 size: ", len(list1))
    print("list2 size: ", len(list2))
    pred_result = []
    sum = 0
    epoch = 20
    for i in range(epoch):
        res = classify(list1, list2, i)
        sum = sum+res
        pred_result.append(res)
    print()
    print("overall precision:", sum/epoch)
    print(pred_result[0:10])
