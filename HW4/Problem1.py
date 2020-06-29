#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 17:03:43 2020

@author: zheng
"""

import numpy as np
from perceptronSeqGD import perceptronLearningSequentialGD
import matplotlib.pyplot as plt
from plotDecBoundaries_HW4 import plotDecBoundaries

train_set_1 = np.loadtxt('/home/zheng/EE559/HW_4/synthetic1_train.csv', delimiter=',')
train_data_1 = train_set_1[:, :-1]
train_label_1 = train_set_1[:, -1]
test_set_1 = np.loadtxt('/home/zheng/EE559/HW_4/synthetic1_test.csv', delimiter=',')
test_data_1 = test_set_1[:, :-1]
test_label_1 = test_set_1[:, -1]
w_final_1, pred_label_1, precision_train_1 = perceptronLearningSequentialGD(train_data_1, train_label_1, train_data_1, train_label_1)
w_final_1, pred_label_1, precision_test_1 = perceptronLearningSequentialGD(train_data_1, train_label_1, test_data_1, test_label_1)
print('Final w on dataset 1 is', w_final_1)
print('Error rate on training set 1 is', 1 - precision_train_1)
print('Error rate on test set 1 is', 1 - precision_test_1)
plotDecBoundaries(train_data_1, train_label_1, w_final_1)

train_set_2 = np.loadtxt('/home/zheng/EE559/HW_4/synthetic2_train.csv', delimiter=',')
train_data_2 = train_set_2[:, :-1]
train_label_2 = train_set_2[:, -1]
test_set_2 = np.loadtxt('/home/zheng/EE559/HW_4/synthetic2_test.csv', delimiter=',')
test_data_2 = test_set_2[:, :-1]
test_label_2 = test_set_2[:, -1]
w_final_2, pred_label_2, precision_train_2 = perceptronLearningSequentialGD(train_data_2, train_label_2, train_data_2, train_label_2)
w_final_2, pred_label_2, precision_test_2 = perceptronLearningSequentialGD(train_data_2, train_label_2, test_data_2, test_label_2)
print('Final w on dataset 2 is', w_final_2)
print('Error rate on training set 2 is', 1 - precision_train_2)
print('Error rate on test set 2 is', 1 - precision_test_2)
plotDecBoundaries(train_data_2, train_label_2, w_final_2)

# plt.figure()
# plt.scatter(train_data_2[np.where(train_label_2 == 1), 0], train_data_2[np.where(train_label_2 == 1), 1])
# plt.scatter(train_data_2[np.where(train_label_2 == 2), 0], train_data_2[np.where(train_label_2 == 2), 1])

train_data_3 = np.loadtxt('/home/zheng/EE559/HW_4/feature_train.csv', delimiter=',')
train_label_3 = np.loadtxt('/home/zheng/EE559/HW_4/label_train.csv', delimiter=',')
test_data_3 = np.loadtxt('/home/zheng/EE559/HW_4/feature_test.csv', delimiter=',')
test_label_3 = np.loadtxt('/home/zheng/EE559/HW_4/label_test.csv', delimiter=',')
w_final_3, pred_label_3, precision_train_3 = perceptronLearningSequentialGD(train_data_3, train_label_3, train_data_3, train_label_3)
w_final_3, pred_label_3, precision_test_3 = perceptronLearningSequentialGD(train_data_3, train_label_3, test_data_3, test_label_3)
print('Final w on dataset 3 is', w_final_3)
print('Error rate on training set 3 is', 1 - precision_train_3)
print('Error rate on test set 3 is', 1 - precision_test_3)
plotDecBoundaries(train_data_3, train_label_3, w_final_3)
# w_final, pred_label, precision = perceptronLearningSequentialGD(train_data, train_label, train_data, train_label)
# plt.figure()
# plt.scatter(train_data[np.where(train_label == 1), 0], train_data[np.where(train_label == 1), 1])
# plt.scatter(train_data[np.where(train_label == 2), 0], train_data[np.where(train_label == 2), 1])