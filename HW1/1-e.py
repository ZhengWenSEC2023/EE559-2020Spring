# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 23:31:52 2020

@author: Zheng Wen
"""

import numpy as np
from KNN import nearestMeansClassifier
from KNN import errorRateCal
from Combination import Combination

combinations = Combination().combine(13,2)
train_set = np.loadtxt("D:\EE559\HW_1\python3\wine_train.csv",delimiter=",")
test_set = np.loadtxt("D:\EE559\HW_1\python3\wine_test.csv",delimiter=",")

train_label = train_set[:,-1]
test_label = test_set[:,-1]
error_rate_test = []
error_rate_train = []
max_error_rate_test, max_error_rate_train = 0, 0
min_error_rate_test, min_error_rate_train = 100, 100

for i in range(len(combinations)):
    combination = np.array(combinations[i]) - 1
    train_data = train_set[:, combination]
    test_data = test_set[:, combination]
    means, pred_train = nearestMeansClassifier(train_data, train_label, train_data)
    means, pred_test = nearestMeansClassifier(train_data, train_label, test_data)
    
    err_test = errorRateCal(pred_test, test_label)
    error_rate_test.append(err_test)
    if err_test < min_error_rate_test:
        min_error_rate_test = err_test
    if err_test > max_error_rate_test:
        max_error_rate_test = err_test
        
    err_train = errorRateCal(pred_train, train_label)
    error_rate_train.append(err_train)
    if err_train < min_error_rate_train:
        min_error_rate_train = err_train
    if err_train > max_error_rate_train:
        max_error_rate_train = err_train
    
error_rate_test = np.array(error_rate_test)
print()
print("For test set:")
print("\tThe standard deviation is", np.std(error_rate_test))
print("\tThe maximum error rate is", max_error_rate_test)
print("\tThe minimum error rate is", min_error_rate_test)
print("\tThe average error rate is", np.average(error_rate_test))

error_rate_train = np.array(error_rate_train)
print()
print("For train set:")
print("\tThe standard deviation is", np.std(error_rate_train))
print("\tThe maximum error rate is", max_error_rate_train)
print("\tThe minimum error rate is", min_error_rate_train)
print("\tThe average error rate is", np.average(error_rate_train))
