# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cityblock
from scipy.spatial.distance import euclidean

# for KL divergence: https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.stats.entropy.html
from scipy.stats import entropy 

def chisquaredist(array1, array2):
    if len(array1) != len(array2):
        raise ValueError('Input arrays need to be the same size.')
    return sum([ ( (array1[i] - array2[i]) ** 2 ) / (array1[i] + array2[i]) for i in range(len(array1))])

train_images = pd.read_excel('../data/hist_rep.xlsx').values
train_labels = pd.read_excel('../data/Training.xlsx').sort_values(['Filename']).values

true_labels = {}
for item in train_labels:
    true_labels[item[0]] = item[1]
    
test_images = np.load('../data/test_SIFT_features.npy')[()] 
test_lab = pd.read_excel('../data/Testing.xlsx').sort_values(['Filename']).values

test_labels = {}
for item in test_lab:
    test_labels[item[0]] = item[1]
    

centers = np.load('../data/cluster_centers.npy')

images = list(test_images.keys())

test_hist = []

for img in images:
    to_add = [img]
    temp_feature = np.zeros(9)
    img_feat = test_images[img]
    for f in img_feat:
        temp_sim = []
        for center in centers:
            temp_sim.append(euclidean(f, center))
        closest_feat_ind = temp_sim.index(max(temp_sim))
        temp_feature[closest_feat_ind] += 1
    to_add += list(temp_feature)
    test_hist.append(to_add)

test_hist_df = pd.DataFrame(test_hist)
writer = pd.ExcelWriter('../data/test_hist_rep.xlsx')
test_hist_df.to_excel(writer, 'sheet1')
writer.save()

test_train_similarity = {}
for test_image_hist in test_hist:
    test_train_similarity[test_image_hist[0]] = {}
    test_hist_1 = list(test_image_hist[1:])
    for train_image_hist in train_images:
        train_hist_2 =list(train_image_hist[1:])
        test_train_similarity[test_image_hist[0]][train_image_hist[0]] = {}
        test_train_similarity[test_image_hist[0]][train_image_hist[0]]["L1"] = cityblock(test_hist_1, train_hist_2)
        test_train_similarity[test_image_hist[0]][train_image_hist[0]]["KL"] = entropy(test_hist_1, train_hist_2)
        test_train_similarity[test_image_hist[0]][train_image_hist[0]]["chi"]  = chisquaredist(test_hist_1, train_hist_2)

json.dump(test_train_similarity, open('../data/test_train_similarity.json', 'w'))