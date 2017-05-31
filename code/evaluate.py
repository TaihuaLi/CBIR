#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 23:57:32 2017

@author: taihuali
"""

'''
This file evaluates the retreival results
Precision, Recall
'''

import pandas as pd
import json

train_labels = pd.read_excel('../data/Training.xlsx').sort_values(['Filename']).values

true_labels = {}
for item in train_labels:
    true_labels[item[0]] = item[1]

test_lab = pd.read_excel('../data/Testing.xlsx').sort_values(['Filename']).values

test_labels = {}
for item in test_lab:
    test_labels[item[0]] = item[1]


test_train_similarity = json.load(open('../data/test_train_similarity.json'))

# KL divergence
KL_results = []
L1_results = []
chi_results = []
for test_img in test_train_similarity.keys():
    temp = [test_img]
    temp2 = [test_img]
    temp3 = [test_img]
    for train_img in test_train_similarity[test_img].keys():
        pair = (test_train_similarity[test_img][train_img]['KL'], train_img)
        pair2 = (test_train_similarity[test_img][train_img]['L1'], train_img)
        pair3 = (test_train_similarity[test_img][train_img]['chi'], train_img)
        temp.append(pair)
        temp2.append(pair2)
        temp3.append(pair3)
    KL_results.append(temp)
    L1_results.append(temp2)
    chi_results.append(temp3)
    
for i in range(len(KL_results)):
    KL_results[i] = [KL_results[i][0]] + sorted(KL_results[i][1:], reverse=False)
    L1_results[i] = [L1_results[i][0]] + sorted(L1_results[i][1:], reverse=False)
    chi_results[i] = [chi_results[i][0]] + sorted(chi_results[i][1:], reverse=False)

true_labels[KL_results[0][2][1]]
test_labels[KL_results[0][0]]

KL_retri_labels = []
L1_retri_labels = []
chi_retri_labels = []
for i in range(len(KL_results)):
    temp = [test_labels[KL_results[i][0]]]
    temp2 = [test_labels[L1_results[i][0]]]
    temp3 = [test_labels[chi_results[i][0]]]
    for j in range(1, len(KL_results[i][1:])):
        if KL_results[i][j][0] != float('inf'):
            temp.append(true_labels[KL_results[i][j][1]])
        if L1_results[i][j][0] != float('inf'):
            temp2.append(true_labels[L1_results[i][j][1]])
        if chi_results[i][j][0] != float('inf'):
            temp3.append(true_labels[chi_results[i][j][1]])
    KL_retri_labels.append(temp)
    L1_retri_labels.append(temp2)
    chi_retri_labels.append(temp3)
    
KL_precision = []
KL_recall = []
for result in KL_retri_labels:
    tp = 0
    groundtruth = result[0]
    temp_pre = [groundtruth]
    temp_rec = [groundtruth]
    for i in range(len(result[1:])):
        if result[1:][i] == groundtruth:
            tp += 1
        else:
            pass
        temp_pre.append(tp/(i+1))
        temp_rec.append(tp/15)
    KL_precision.append(temp_pre)
    KL_recall.append(temp_rec)
KL_precision = pd.DataFrame(KL_precision)
KL_recall = pd.DataFrame(KL_recall)
    
L1_precision = []
L1_recall = []
for result in L1_retri_labels:
    tp = 0
    groundtruth = result[0]
    temp_pre = [groundtruth]
    temp_rec = [groundtruth]
    for i in range(len(result[1:])):
        if result[1:][i] == groundtruth:
            tp += 1
        else:
            pass
        temp_pre.append(tp/(i+1))
        temp_rec.append(tp/15)
    L1_precision.append(temp_pre)
    L1_recall.append(temp_rec)
L1_precision = pd.DataFrame(L1_precision)
L1_recall = pd.DataFrame(L1_recall)
    
    
chi_precision = []
chi_recall = []
for result in chi_retri_labels:
    tp = 0
    groundtruth = result[0]
    temp_pre = [groundtruth]
    temp_rec = [groundtruth]
    for i in range(len(result[1:])):
        if result[1:][i] == groundtruth:
            tp += 1
        else:
            pass
        temp_pre.append(tp/(i+1))
        temp_rec.append(tp/15)
    chi_precision.append(temp_pre)
    chi_recall.append(temp_rec)
chi_precision = pd.DataFrame(chi_precision)
chi_recall = pd.DataFrame(chi_recall)

KL_writer_1 = pd.ExcelWriter('../eval/KL_precision.xlsx')
KL_writer_2 = pd.ExcelWriter('../eval/KL_recall.xlsx')
KL_precision.to_excel(KL_writer_1, 'sheet1')
KL_recall.to_excel(KL_writer_2, 'sheet1')
KL_writer_1.save()
KL_writer_2.save()

L1_writer_1 = pd.ExcelWriter('../eval/L1_precision.xlsx')
L1_writer_2 = pd.ExcelWriter('../eval/L1_recall.xlsx')
L1_precision.to_excel(L1_writer_1, 'sheet1')
L1_recall.to_excel(L1_writer_2, 'sheet1')
L1_writer_1.save()
L1_writer_2.save()

chi_writer_1 = pd.ExcelWriter('../eval/chi_precision.xlsx')
chi_writer_2 = pd.ExcelWriter('../eval/chi_recall.xlsx')
chi_precision.to_excel(chi_writer_1, 'sheet1')
chi_recall.to_excel(chi_writer_2, 'sheet1')
chi_writer_1.save()
chi_writer_2.save()


KL_precision = KL_precision.fillna(0)
KL_recall = KL_recall.fillna(0)






