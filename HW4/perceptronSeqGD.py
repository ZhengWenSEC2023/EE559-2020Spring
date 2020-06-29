#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 22:30:46 2020

@author: zheng
"""
import sys
import numpy as np

def perceptronLearningSequentialGD(train_data, train_label, test_data, test_label):
    
    np.random.seed(seed=23)
    train_set = np.concatenate((train_data, train_label[:, None]), axis=1)
    train_set = train_set[np.random.permutation(np.shape(train_set)[0]), :]
    y = train_set[:, -1]
    X = train_set[:, :-1]
    X[np.where(y == 2), :] *= -1
    aug_train = np.zeros(np.shape(y))
    aug_train[np.where(y == 1)] = 1
    aug_train[np.where(y == 2)] = -1
    
    X_aug = np.concatenate((aug_train[:, None], X), axis=1)
    
    
    w = 0.1 * np.array([1, 1, 1])
    eta = 1
    
    iter = 1000
    for i in range(iter):
        min_J = sys.maxsize
        for j in range(np.shape(X_aug)[0]):
            if w.T @ X_aug[j] < 0:
                w += eta * X_aug[j]
            if i == iter - 1:
                for k in range(np.shape(X_aug)[0]):
                    J = 0
                    if w.T @ X_aug[k] <= 0:
                        J += -(w.T @ X_aug[k])
                if J < min_J:
                    w_final = w
                    min_J = J
    
    
    
    aug_test = np.zeros(np.shape(test_label)) + 1
    test_data_aug = np.concatenate((aug_test[:, None], test_data), axis=1)
    pred_label = np.zeros(np.shape(test_label))
    for i in range(np.shape(test_data_aug)[0]):
        if w.T @ test_data_aug[i] >= 0:
            pred_label[i] = 1
        else:
            pred_label[i] = 2
    precision = np.sum(pred_label == test_label) / np.shape(test_label)
    return w_final, pred_label, precision