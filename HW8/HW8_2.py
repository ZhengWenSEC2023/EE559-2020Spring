# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:21:44 2020

@author: Lenovo
"""

import os.path as osp
import numpy as np
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

data_path = 'D:\\EE559\\HW_8\\wine_csv'
train_data_name = 'feature_train.csv'
train_label_name = 'label_train.csv'

train_data = np.loadtxt(osp.join(data_path, train_data_name), delimiter=",")[:, :2]
train_label = np.loadtxt(osp.join(data_path, train_label_name), delimiter=",")

# (a)
skf = StratifiedKFold(n_splits=5)
clf = svm.SVC(C=1, kernel='rbf', gamma=1)
acc_a = []
for train_index, val_index in skf.split(train_data, train_label):
    X_train, X_val = train_data[train_index], train_data[val_index]
    y_train, y_val = train_label[train_index], train_label[val_index]
    clf.fit(X_train, y_train)
    pred = clf.predict(X_val)
    acc_a.append(np.sum(pred == y_val) / len(y_val))
print('average cross validation accuracy is', np.mean(acc_a))

#(b)
size_C = 50
size_gamma = 50
C = np.logspace(-3, 3, num=size_C)
gamma = np.logspace(-3, 3, num=size_gamma)
ACC = np.zeros((size_gamma, size_C))
DEV = np.zeros((size_gamma, size_C))
for i in range(size_gamma):
    for j in range(size_C):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
        clf = svm.SVC(C=C[j], kernel='rbf', gamma=gamma[i])
        acc = []
        for train_index, val_index in skf.split(train_data, train_label):
            X_train, X_val = train_data[train_index], train_data[val_index]
            y_train, y_val = train_label[train_index], train_label[val_index]
            clf.fit(X_train, y_train)
            pred = clf.predict(X_val)
            acc.append(np.sum(pred == y_val) / len(y_val))
        ACC[i, j] = np.mean(acc)
        DEV[i, j] = np.std(acc)

plt.imshow(ACC)
plt.colorbar()

max_pos = np.where(ACC==np.amax(ACC))
if np.size(max_pos) != 2:
    num_same = len(max_pos[0])
    min_std = np.Inf
    for i in range(num_same):
        if DEV[max_pos[0][i], max_pos[1][i]] < min_std:
            min_std = DEV[max_pos[0][i], max_pos[1][i]]
            temp = (max_pos[0][i], max_pos[1][i])
    final_pos = temp
else:
    final_pos = max_pos

print('Max average accuracy is', ACC[final_pos])
print('Corresponding standard deviation is', DEV[final_pos])
print('Corresponding gamma is', gamma[final_pos[0]])
print('Corresponding C is', C[final_pos[1]])

print()
#(c)
size_C = 50
size_gamma = 50
time = 20
C = np.logspace(-3, 3, num=size_C)
gamma = np.logspace(-3, 3, num=size_gamma)
ACC_c = np.zeros((size_gamma, size_C, time))
DEV_c = np.zeros((size_gamma, size_C, time))
for t in range(time):
    for i in range(size_gamma):
        for j in range(size_C):
            skf = StratifiedKFold(n_splits=5, shuffle=True)
            clf = svm.SVC(C=C[j], kernel='rbf', gamma=gamma[i])
            acc = []
            for train_index, val_index in skf.split(train_data, train_label):
                X_train, X_val = train_data[train_index], train_data[val_index]
                y_train, y_val = train_label[train_index], train_label[val_index]
                clf.fit(X_train, y_train)
                pred = clf.predict(X_val)
                acc.append(np.sum(pred == y_val) / len(y_val))
            ACC_c[i, j, t] = np.mean(acc)
            DEV_c[i, j, t] = np.std(acc)
    max_pos = np.where(ACC_c[:, :, t]==np.amax(ACC_c[:, :, t]))
    if np.size(max_pos) != 2:
        num_same = len(max_pos[0])
        min_std = np.Inf
        for i in range(num_same):
            if DEV_c[max_pos[0][i], max_pos[1][i], t] < min_std:
                min_std = DEV_c[max_pos[0][i], max_pos[1][i], t]
                temp = (max_pos[0][i], max_pos[1][i])
        final_pos = temp
    else:
        final_pos = max_pos
    print(t, 'gamma, C =', gamma[final_pos[0]], C[final_pos[1]])
    

final_ACC = np.mean(ACC_c, axis=2)
final_DEV = np.mean(DEV_c, axis=2)
max_pos = np.where(final_ACC==np.amax(final_ACC))
if np.size(max_pos) != 2:
    num_same = len(max_pos[0])
    min_std = np.Inf
    for i in range(num_same):
        if final_DEV[max_pos[0][i], max_pos[1][i]] < min_std:
            min_std = final_DEV[max_pos[0][i], max_pos[1][i]]
            temp = (max_pos[0][i], max_pos[1][i])
    final_pos = temp
else:
    final_pos = max_pos

print('Max average accuracy is', final_ACC[final_pos])
print('Corresponding standard deviation is', final_DEV[final_pos])
print('Corresponding gamma is', gamma[final_pos[0]])
print('Corresponding C is', C[final_pos[1]])

# d
test_data_name = 'feature_test.csv'
test_label_name = 'label_test.csv'
test_data = np.loadtxt(osp.join(data_path, test_data_name), delimiter=",")[:, :2]
test_label = np.loadtxt(osp.join(data_path, test_label_name), delimiter=",")
skf = StratifiedKFold(n_splits=5)
clf = svm.SVC(C=C[final_pos[1]], kernel='rbf', gamma=gamma[final_pos[0]])
clf.fit(train_data, train_label)
pred = clf.predict(test_data)

acc = np.sum(pred == test_label) / len(test_label)
print('test set accuracy is', acc)
print('# of std is', np.abs(acc - final_ACC[final_pos]) / final_DEV[final_pos])