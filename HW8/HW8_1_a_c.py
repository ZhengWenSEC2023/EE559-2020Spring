# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 22:21:44 2020

@author: Lenovo
"""

import os.path as osp
import numpy as np
from sklearn import svm
from plotSVMBoundaries import plotSVMBoundaries

data_path = 'D:\\EE559\\HW_8\\HW8_1_csv'
train_data_name = 'train_x.csv'
train_label_name = 'train_y.csv'

train_data = np.loadtxt(osp.join(data_path, train_data_name), delimiter=",")
train_label = np.loadtxt(osp.join(data_path, train_label_name), delimiter=",")
# C = 1
clf = svm.SVC(C=1, kernel='linear')
clf.fit(train_data, train_label)
plotSVMBoundaries(train_data, train_label, clf, clf.support_vectors_)
pred = clf.predict(train_data)
acc = np.sum(pred == train_label) / len(train_label)
print('C = 1, accuracy is', acc)

# C = 100
clf = svm.SVC(C=100, kernel='linear')
clf.fit(train_data, train_label)
plotSVMBoundaries(train_data, train_label, clf, clf.support_vectors_)
pred = clf.predict(train_data)
acc = np.sum(pred == train_label) / len(train_label)

print("C = 100, accuracy is", acc)
print('Support Vectors are', clf.support_vectors_)
print('w =', clf.coef_)
print('b =', clf.intercept_)

print(clf.decision_function(clf.support_vectors_))

# C = 10000
clf = svm.SVC(C=10000, kernel='linear')
clf.fit(train_data, train_label)
plotSVMBoundaries(train_data, train_label, clf, clf.support_vectors_)
pred = clf.predict(train_data)
acc = np.sum(pred == train_label) / len(train_label)
print('Support Vectors are', clf.support_vectors_)
print(clf.decision_function(clf.support_vectors_))