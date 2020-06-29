# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:24:05 2020

@author: Lenovo
"""

import os.path as osp
import numpy as np
from sklearn import svm
from plotSVMBoundaries import plotSVMBoundaries

data_path = 'D:\\EE559\\HW_8\\HW8_2_csv'
train_data_name = 'train_x.csv'
train_label_name = 'train_y.csv'

train_data = np.loadtxt(osp.join(data_path, train_data_name), delimiter=",")
train_label = np.loadtxt(osp.join(data_path, train_label_name), delimiter=",")

# C = 50, gamma = default
clf_50_default = svm.SVC(C=50, kernel='rbf', gamma='auto')
clf_50_default.fit(train_data, train_label)
plotSVMBoundaries(train_data, train_label, clf_50_default)
pred = clf_50_default.predict(train_data)
acc = np.sum(pred == train_label) / len(train_label)
print('C = 50, gamma = default, accuracy is', acc)

# C = 5000, gamma = default
clf_5000_default = svm.SVC(C=5000, kernel='rbf', gamma='auto')
clf_5000_default.fit(train_data, train_label)
plotSVMBoundaries(train_data, train_label, clf_5000_default)
pred = clf_5000_default.predict(train_data)
acc = np.sum(pred == train_label) / len(train_label)
print("C = 5000, gamma = default, accuracy is", acc)

# C = default, gamma = 10
clf_default_10 = svm.SVC(kernel='rbf', gamma=10)
clf_default_10.fit(train_data, train_label)
plotSVMBoundaries(train_data, train_label, clf_default_10)
pred = clf_default_10.predict(train_data)
acc = np.sum(pred == train_label) / len(train_label)
print('C = default, gamma = 10, accuracy is', acc)

# C = default, gamma = 50
clf_default_50 = svm.SVC(kernel='rbf', gamma=50)
clf_default_50.fit(train_data, train_label)
plotSVMBoundaries(train_data, train_label, clf_default_50)
pred = clf_default_50.predict(train_data)
acc = np.sum(pred == train_label) / len(train_label)
print('C = default, gamma = 50, accuracy is', acc)

# C = default, gamma = 50
clf_default_5000 = svm.SVC(kernel='rbf', gamma=500)
clf_default_5000.fit(train_data, train_label)
plotSVMBoundaries(train_data, train_label, clf_default_5000)
pred = clf_default_5000.predict(train_data)
acc = np.sum(pred == train_label) / len(train_label)
print('C = default, gamma = 500, accuracy is', acc)