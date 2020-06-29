# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 16:41:12 2020

@author: Zheng Wen
"""

import numpy as np


def nearestMeansClassifier(train_data, train_labels, test_data):
    train_data_size = np.zeros((np.size(np.unique(train_labels)), 2))
    train_data_size[:,-1] = np.unique(train_labels)
    means = np.zeros((np.size(np.unique(train_labels)), np.shape(train_data)[1] + 1))
    means[:,-1] = np.unique(train_labels)
        
    for i in range(np.shape(train_data)[0]):
        means[np.where(means[:,-1] == train_labels[i]),0:-1] += train_data[i,:]
        train_data_size[np.where(train_data_size[:, -1] == train_labels[i]), 0] += 1
    for i in range(np.shape(means)[0]):
        means[i,0:-1] /= train_data_size[i, 0]
    
    predict = np.zeros((np.shape(test_data)[0], 1))
    
    tag = np.unique(train_labels)
    
    for i in range(np.shape(test_data)[0]):
        dist = np.zeros((np.size(np.unique(train_labels)), 1))
        for j in range(np.size(np.unique(train_labels))):
            dist[j,0] = np.sqrt(np.sum(np.square(means[j,0:-1] - test_data[i,:])))
        predict[i,0] = tag[np.argmin(dist)]
        
    return means, predict


def errorRateCal(predict, actual):
    total = np.size(predict)
    error = 0
    for i in range(np.size(predict)):
        if predict[i] != actual[i]:
            error += 1
    
    return error / total * 100