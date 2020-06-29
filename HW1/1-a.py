# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 16:41:12 2020

@author: Zheng Wen
"""

import numpy as np
from plotDecBoundaries import plotDecBoundaries
from KNN import nearestMeansClassifier
from KNN import errorRateCal


train_set_1 = np.loadtxt("D:\EE559\HW_1\python3\synthetic1_train.csv",delimiter=",")
train_data_1 = train_set_1[:, 0:-1]
train_label_1 = train_set_1[:,-1]

test_set_1 = np.loadtxt("D:\EE559\HW_1\python3\synthetic1_test.csv",delimiter=",")
test_data_1 = test_set_1[:, 0:-1]
test_label_1 = test_set_1[:,-1]

means_1, pred_train_1 = nearestMeansClassifier(train_data_1, train_label_1, train_data_1)
means_1, pred_test_1 = nearestMeansClassifier(train_data_1, train_label_1, test_data_1)
plotDecBoundaries(train_data_1, train_label_1, means_1[:, 0:-1])
print("Error Rate of synthetic1 training set =" + str(errorRateCal(pred_train_1, train_label_1)))
print("Error Rate of synthetic1 test set =" + str(errorRateCal(pred_test_1, test_label_1)))


train_set_2 = np.loadtxt("D:\EE559\HW_1\python3\synthetic2_train.csv",delimiter=",")
train_data_2 = train_set_2[:, 0:-1]
train_label_2 = train_set_2[:,-1]

test_set_2 = np.loadtxt("D:\EE559\HW_1\python3\synthetic2_test.csv",delimiter=",")
test_data_2 = test_set_2[:, 0:-1]
test_label_2 = test_set_2[:,-1]

means_2, pred_train_2 = nearestMeansClassifier(train_data_2, train_label_2, train_data_2)
means_2, pred_test_2 = nearestMeansClassifier(train_data_2, train_label_2, test_data_2)
plotDecBoundaries(train_data_2, train_label_2, means_2[:, 0:-1])
print("Error Rate of synthetic2 training set =" + str(errorRateCal(pred_train_2, train_label_2)))
print("Error Rate of synthetic2 test set =" + str(errorRateCal(pred_test_2, test_label_2)))
