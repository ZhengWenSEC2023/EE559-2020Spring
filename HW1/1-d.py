# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 19:23:16 2020

@author: Zheng Wen
"""

import numpy as np
from plotDecBoundaries import plotDecBoundaries
from KNN import nearestMeansClassifier
from KNN import errorRateCal
from Combination import Combination

combinations = Combination().combine(13,2)
train_set = np.loadtxt("D:\EE559\HW_1\python3\wine_train.csv",delimiter=",")
test_set = np.loadtxt("D:\EE559\HW_1\python3\wine_test.csv",delimiter=",")

min_error = 100
min_comb = 0

train_label = train_set[:,-1]
test_label = test_set[:,-1]

for i in range(len(combinations)):
    combination = np.array(combinations[i]) - 1
    train_data = train_set[:, combination]
    test_data = test_set[:, combination]
    means, pred_train = nearestMeansClassifier(train_data, train_label, train_data)
    means, pred_test = nearestMeansClassifier(train_data, train_label, test_data)
    error_rate = errorRateCal(pred_train, train_label)
    
    if error_rate < min_error:
        min_comb = np.array(combinations[i])
        min_error = error_rate
        min_means = means.copy()
        min_pred_train = pred_train.copy()
        min_pred_test = pred_test.copy()
    
plotDecBoundaries(train_set[:, (min_comb - 1)], train_set[:,-1], min_means[:, 0:-1])
print("The feature chosen are:" + str(min_comb - 1))
print("Minimum error rate of wine training set =" + str(errorRateCal(min_pred_train, train_label)))
print("Minimum error rate of wine test set =" + str(errorRateCal(min_pred_test, test_label)))