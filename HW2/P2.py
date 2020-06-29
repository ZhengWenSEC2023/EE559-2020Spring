# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 22:57:47 2020

@author: Lenovo
"""

import numpy as np
from plotDecBoundaries import plotDecBoundariesDouble, plotDecBoundariesMul
from knn import OVRClassfier
from knn import errorRateCal


train_set = np.loadtxt("D:\EE559\HW_1\python3\wine_train.csv",delimiter=",")

test_set = np.loadtxt("D:\EE559\HW_1\python3\wine_test.csv",delimiter=",")

means, predict_tests, predict_trains, train_labels, train_datas, test_labels, final_pred_train, final_pred_test = OVRClassfier(train_set, test_set)

print('Error rate on training set is', errorRateCal(final_pred_train, train_set[:,-1]))
print('Error rate on test set is', errorRateCal(final_pred_test, test_set[:,-1]))

for i in range(3):
    plotDecBoundariesDouble(train_datas[i], train_labels[i], means[i][:, 0:-1])

plotDecBoundariesMul(train_set)