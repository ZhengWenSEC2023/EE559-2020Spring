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


def OVRClassfier(train_set, test_set):
    labels_rest = np.array([[3],[2],[1]])
    labels_resetto = np.array([[2],[1],[3]])
    means = []
    train_labels = []
    train_datas = []
    predict_test = []
    predict_train = []
    test_labels = []
    train_data = train_set[:, 0:2]
    train_label = train_set[:, -1]
    test_data = test_set[:, 0:2]
    test_label = test_set[:, -1]
    
    for i in range(3):
        train_sublabel = train_label.copy()
        test_sublabel = test_label.copy()
        train_sublabel[train_sublabel == labels_rest[i]] = labels_resetto[i]
        test_sublabel[test_sublabel == labels_rest[i]] = labels_resetto[i]
        train_datas.append(train_data)
        train_labels.append(train_sublabel)
        test_labels.append(test_sublabel)
        
        mean, pred_train = nearestMeansClassifier(train_data, train_sublabel, train_data)
        mean, pred_test = nearestMeansClassifier(train_data, train_sublabel, test_data)
        
        means.append(mean)
        predict_test.append(pred_test)
        predict_train.append(pred_train)

    final_pred_train = np.zeros(np.size(predict_train[1]))
    final_pred_test = np.zeros(np.size(predict_test[1]))
    
    for i in range(np.size(predict_train[0])):
        if predict_train[0][i] == 1 and predict_train[1][i] != 3 and predict_train[2][i] != 2:
            final_pred_train[i] = 1
        elif predict_train[0][i] != 1 and predict_train[1][i] == 3 and predict_train[2][i] != 2:
            final_pred_train[i] = 3
        elif predict_train[0][i] != 1 and predict_train[1][i] != 3 and predict_train[2][i] == 2:
            final_pred_train[i] = 2
        else:
            final_pred_train[i] = 0
            
    for i in range(np.size(predict_test[0])):
        if predict_test[0][i] == 1 and predict_test[1][i] != 3 and predict_test[2][i] != 2:
            final_pred_test[i] = 1
        elif predict_test[0][i] != 1 and predict_test[1][i] == 3 and predict_test[2][i] != 2:
            final_pred_test[i] = 3
        elif predict_test[0][i] != 1 and predict_test[1][i] != 3 and predict_test[2][i] == 2:
            final_pred_test[i] = 2
        else:
            final_pred_test[i] = 0
    
    return means, predict_test, predict_train, train_labels, train_datas, test_labels, final_pred_train, final_pred_test


def OVRClassfierPlot(train_set, test_data):
    labels_rest = np.array([[3],[2],[1]])
    labels_resetto = np.array([[2],[1],[3]])
    means = []
    train_labels = []
    train_datas = []
    predict_test = []
    predict_train = []
    train_data = train_set[:, 0:2]
    train_label = train_set[:, -1]
    
    for i in range(3):
        train_sublabel = train_label.copy()
        train_sublabel[train_sublabel == labels_rest[i]] = labels_resetto[i]
        train_datas.append(train_data)
        train_labels.append(train_sublabel)        
        mean, pred_train = nearestMeansClassifier(train_data, train_sublabel, train_data)
        mean, pred_test = nearestMeansClassifier(train_data, train_sublabel, test_data)        
        means.append(mean)
        predict_test.append(pred_test)
        predict_train.append(pred_train)

    final_pred_train = np.zeros(np.size(predict_train[1]))
    final_pred_test = np.zeros(np.size(predict_test[1]))
    
    for i in range(np.size(predict_train[0])):
        if predict_train[0][i] == 1 and predict_train[1][i] != 3 and predict_train[2][i] != 2:
            final_pred_train[i] = 1
        elif predict_train[0][i] != 1 and predict_train[1][i] == 3 and predict_train[2][i] != 2:
            final_pred_train[i] = 3
        elif predict_train[0][i] != 1 and predict_train[1][i] != 3 and predict_train[2][i] == 2:
            final_pred_train[i] = 2
        else:
            final_pred_train[i] = 0
            
    for i in range(np.size(predict_test[0])):
        if predict_test[0][i] == 1 and predict_test[1][i] != 3 and predict_test[2][i] != 2:
            final_pred_test[i] = 1
        elif predict_test[0][i] != 1 and predict_test[1][i] == 3 and predict_test[2][i] != 2:
            final_pred_test[i] = 3
        elif predict_test[0][i] != 1 and predict_test[1][i] != 3 and predict_test[2][i] == 2:
            final_pred_test[i] = 2
        else:
            final_pred_test[i] = 0
    
    return final_pred_test