# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 19:05:37 2020

@author: Zheng Wen
"""

import numpy as np
from plotDecBoundaries import plotDecBoundaries
from KNN import nearestMeansClassifier
from KNN import errorRateCal


train_set = np.loadtxt("D:\EE559\HW_1\python3\wine_train.csv",delimiter=",")
train_data = train_set[:, 0:2]
train_label = train_set[:,-1]

test_set = np.loadtxt("D:\EE559\HW_1\python3\wine_test.csv",delimiter=",")
test_data = test_set[:, 0:2]
test_label = test_set[:,-1]

means, pred_train = nearestMeansClassifier(train_data, train_label, train_data)
means, pred_test = nearestMeansClassifier(train_data, train_label, test_data)
plotDecBoundaries(train_data, train_label, means[:, 0:-1])
print("Error Rate of wine training set =" + str(errorRateCal(pred_train, train_label)))
print("Error Rate of wine test set =" + str(errorRateCal(pred_test, test_label)))