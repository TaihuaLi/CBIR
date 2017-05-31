#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 21:23:52 2017

@author: taihuali
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

centers = np.load('../data/cluster_centers.npy')

features = np.load('../data/train_SIFT_features.npy')[()]

images = list(features.keys())

image_hist = []

for img in images:
    to_add = [img]
    temp_feature = np.zeros(9)
    img_feat = features[img]
    for f in img_feat:
        temp_sim = []
        for center in centers:
            temp_sim.append(euclidean(f, center))
        closest_feat_ind = temp_sim.index(max(temp_sim))
        temp_feature[closest_feat_ind] += 1
    to_add += list(temp_feature)
    image_hist.append(to_add)
    

hist_rep = pd.DataFrame(image_hist)

writer = pd.ExcelWriter('../data/hist_rep.xlsx')
hist_rep.to_excel(writer, 'sheet1', index=False)
writer.save()
