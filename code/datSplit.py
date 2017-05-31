#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:46:17 2017

@author: taihuali
"""

import random
import numpy as np
import pandas as pd

dat = np.load('../SIFT_features.npy')
labels = pd.read_excel('../Labels.xlsx')

subset_ind = []
start = 0
end = 79
for i in range(0, 1360//80):
    subset_ind += random.sample(range(start, end), 20)
    start += 80
    end += 80

subset = labels.iloc[subset_ind]

test_set = []
start = 0
end = 20
for i in range(0, 1360//80):
    test_set += random.sample(subset_ind[start:end], 5)
    start += 20
    end += 20 

train = labels.iloc[[x for x in subset_ind if x not in test_set]]
test = labels.iloc[test_set]

train_writer = pd.ExcelWriter('../data/Training.xlsx')
test_writer = pd.ExcelWriter('../data/Testing.xlsx')
train.to_excel(train_writer, 'sheet1', index=False)
test.to_excel(test_writer, 'sheet1', index=False)
train_writer.save()
test_writer.save()

features = np.load('../SIFT_features.npy')[()]

train_dat = {}
test_dat = {}
for k, v in features.items():
    if k in train.Filename.values:
        train_dat[k] = v
    elif k in test.Filename.values:
        test_dat[k] = v

np.save('../data/train_SIFT_features.npy', train_dat)
np.save('../data/test_SIFT_features.npy', test_dat)