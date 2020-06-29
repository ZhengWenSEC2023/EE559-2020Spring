# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 21:45:40 2020

@author: Lenovo
"""

import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from collections import Counter

class MSE_binary(LinearRegression):
    def __init__(self):
        # print('Calling newly created MSE binary function...')
        super(MSE_binary, self).__init__()
    def predict(self, X):
        y = self._decision_function(X)
        y_round = np.round(y)
        y_count = Counter(y_round)
        top_2 = y_count.most_common(2)
        label1 = np.max([top_2[0][0], top_2[1][0]])
        label2 = np.min([top_2[0][0], top_2[1][0]])
        thr = (label1 - label2) / 2 + label2
        y_binary = y
        y_binary[np.where(y_binary >= thr)] = label1
        y_binary[np.where(y_binary < thr)] = label2
        return y_binary
        

np.set_printoptions(precision=3, suppress=True)

train_set = np.loadtxt("D:\\EE559\\HW_6\\wine_train.csv",delimiter=",")
train_data = train_set[:, 0:-1]
train_label = train_set[:,-1]

test_set = np.loadtxt("D:\\EE559\\HW_6\\wine_test.csv",delimiter=",")
test_data = test_set[:, 0:-1]
test_label = test_set[:,-1]


## b
train_mean = np.mean(train_data, axis=0)
train_std = np.std(train_data, axis=0)
print('train mean', repr(train_mean))
print('train std', repr(train_std))


nor_train_data = (train_data - train_mean) / train_std
nor_test_data = (test_data - train_mean) / train_std

## d - 1
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(nor_train_data[:, :2], train_label)
train_accuracy = np.sum((train_label == clf.predict(nor_train_data[:, :2])).astype(float)) / np.shape(train_label)[0]
test_accuracy = np.sum((test_label == clf.predict(nor_test_data[:, :2])).astype(float)) / np.shape(test_label)[0]
print('Weight with two Feature', repr(clf.coef_))
print('Classification accuracy on training set with two feature', train_accuracy)
print('Classification accuracy on test set with two feature', test_accuracy)

## d - 2
clf.fit(nor_train_data, train_label)
train_accuracy = np.sum((train_label == clf.predict(nor_train_data)).astype(float)) / np.shape(train_label)[0]
test_accuracy = np.sum((test_label == clf.predict(nor_test_data)).astype(float)) / np.shape(test_label)[0]
print('Weight with all feature', repr(clf.coef_))
print('Classification accuracy on training set with all feature', train_accuracy)
print('Classification accuracy on test set with all feature', test_accuracy)

clf.fit(nor_train_data[:, :2], train_label)
np.random.seed(200)
## e - 1
max_train = 0
for i in range(100):
    co_in = np.random.rand(3, 2)
    # clf.fit(nor_train_data[:, :2], train_label, coef_init=co_in)
    train_accuracy = np.sum((train_label == clf.predict(nor_train_data[:, :2])).astype(float)) / np.shape(train_label)[0]
    test_accuracy = np.sum((test_label == clf.predict(nor_test_data[:, :2])).astype(float)) / np.shape(test_label)[0]
    if train_accuracy >= max_train:
        max_train = train_accuracy
        max_weight = clf.coef_
        max_test = test_accuracy
print('Weight with two Feature', repr(max_weight))
print('Classification accuracy on training set with two feature', max_train)
print('Classification accuracy on test set with two feature', max_test)

np.random.seed(100)
## e- 2
max_train = 0
clf.fit(nor_train_data, train_label)
for i in range(100):
    co_in = np.random.rand(3, 13)
    clf.fit(nor_train_data, train_label, coef_init=co_in)
    train_accuracy = np.sum((train_label == clf.predict(nor_train_data)).astype(float)) / np.shape(train_label)[0]
    test_accuracy = np.sum((test_label == clf.predict(nor_test_data)).astype(float)) / np.shape(test_label)[0]
    if train_accuracy > max_train:
        max_train = train_accuracy
        max_weight = clf.coef_
        max_test = test_accuracy
print('Weight with all Feature', repr(max_weight))
print('Classification accuracy on training set with all feature', max_train)
print('Classification accuracy on test set with all feature', max_test)
print()
## g
train_set_1 = train_set[np.where(train_label == 1)]
train_set_2 = train_set[np.where(train_label == 3)]
train_set_12 = np.concatenate((train_set_1, train_set_2), axis=0)
train_data_12 = train_set_12[:, :-1]
train_label_12 = train_set_12[:, -1]

test_set_1 = test_set[np.where(test_label == 1)]
test_set_2 = test_set[np.where(test_label == 3)]
test_set_12 = np.concatenate((test_set_1, test_set_2), axis=0)
test_data_12 = test_set_12[:, :-1]
test_label_12 = test_set_12[:, -1]

binary_model = MSE_binary()
binary_model.fit(train_data_12, train_label_12)
res = binary_model.predict(test_data_12)

mc_model = OneVsRestClassifier(binary_model)



mc_model.fit(train_data, train_label)
pred_MSE_train = mc_model.predict(train_data)
acc_MSE_train = np.sum((pred_MSE_train == train_label).astype(float)) / np.shape(train_label)[0]
print('Classification accuracy on training set with all features with MSE, unnormalized is', acc_MSE_train)
pred_MSE_test = mc_model.predict(test_data)
acc_MSE_test = np.sum((pred_MSE_test == test_label).astype(float)) / np.shape(test_label)[0]
print('Classification accuracy on test set with all features with MSE, unnormalized is', acc_MSE_test)
print()
mc_model.fit(train_data[:, :2], train_label)
pred_MSE_train = mc_model.predict(train_data[:, :2])
acc_MSE_train = np.sum((pred_MSE_train == train_label).astype(float)) / np.shape(train_label)[0]
print('Classification accuracy on training set with two features with MSE, unnormalized is', acc_MSE_train)
pred_MSE_test = mc_model.predict(test_data[:, :2])
acc_MSE_test = np.sum((pred_MSE_test == test_label).astype(float)) / np.shape(test_label)[0]
print('Classification accuracy on test set with two features with MSE, unnormalized is', acc_MSE_test)
print()
mc_model.fit(nor_train_data, train_label)
pred_MSE_train = mc_model.predict(nor_train_data)
acc_MSE_train = np.sum((pred_MSE_train == train_label).astype(float)) / np.shape(train_label)[0]
print('Classification accuracy on training set with all features with MSE, normalized is', acc_MSE_train)
pred_MSE_test = mc_model.predict(nor_test_data)
acc_MSE_test = np.sum((pred_MSE_test == test_label).astype(float)) / np.shape(test_label)[0]
print('Classification accuracy on test set with all features with MSE, normalized is', acc_MSE_test)
print()
mc_model.fit(nor_train_data[:, :2], train_label)
pred_MSE_train = mc_model.predict(nor_train_data[:, :2])
acc_MSE_train = np.sum((pred_MSE_train == train_label).astype(float)) / np.shape(train_label)[0]
print('Classification accuracy on training set with two features with MSE, normalized is', acc_MSE_train)
pred_MSE_test = mc_model.predict(nor_test_data[:, :2])
acc_MSE_test = np.sum((pred_MSE_test == test_label).astype(float)) / np.shape(test_label)[0]
print('Classification accuracy on test set with two features with MSE, normalized is', acc_MSE_test)

