#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 15:31:45 2017

@author: taihuali
"""

import cv2, os, json
import numpy as np
import pandas as pd

feature_directory = '../../flowers'

image_files = [x for x in os.listdir(feature_directory) if x.endswith('jpg')]

SIFT_features = {}
SIFT_keypoints = {}
sift = cv2.xfeatures2d.SIFT_create()    
for i in image_files:
    image_dir = feature_directory+'/'+i
    grayImage = cv2.cvtColor(cv2.imread(image_dir), cv2.COLOR_BGR2GRAY)
    kp, feat = sift.detectAndCompute(grayImage, None)
    SIFT_features[i] = feat
    SIFT_keypoints[i] = kp
np.save('SIFT_features.npy', SIFT_features)
#np.save('SIFT_keypoints.npy', SIFT_keypoints)
# to load: np.load('SIFT_features.npy')

names = ['Daffodil', 'Snowdrop', 'Lilly Valley', 'Bluebell', 'Crocus', 'Iris',
         'Tigerlily', 'Tulip', 'Fritillary', 'Sunflower', 'Daisy', 'Colts Foot',
         'Dandelion', 'Cowslip', 'Buttercup', 'Windflower', 'Pansy']
n = 80
j = 0
labels = []
for i in image_files:
    if n != 0:
        labels.append(names[j])
        n -= 1
    else:
        n = 80
        j += 1
        labels.append(names[j])
        n -= 1
image_fnames = np.array(image_files)
labels = np.array(labels)
labeled = pd.DataFrame(np.vstack([image_fnames, labels]).T, columns=['Filename', 'Label'])

writer = pd.ExcelWriter('Labels.xlsx')
labeled.to_excel(writer, 'sheet1', index=False)
writer.save()