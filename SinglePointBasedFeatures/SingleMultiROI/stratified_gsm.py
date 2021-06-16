# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Thu May  6 21:59:57 2021
==============================================================================
A.1 Stratified Gray Scale Median
==============================================================================
Inputs:
    - img:              image
    - mask:             2D array with 1 inside ROI [int32]
    - perimeter_lumen:  the perimeter of plaque near lumen
    - percentages:      percentages for stratified GSM
Outputs:
    - features:         Gray Scale Median (GSM)
==============================================================================
"""

import numpy as np
from scipy.spatial.distance import cdist

def get_perc_ROI(mask, perimeter_lumen, perc):
    dist = np.empty(mask.shape)
    dist[:] = np.inf
    II = np.argwhere(mask)
    JJ = np.argwhere(perimeter_lumen)
    K = tuple(II.T)
    dist[K] = cdist(II, JJ).min(axis=1, initial=np.inf)     
    percPixels = np.fix(perc * np.count_nonzero(mask) ).astype('i')
    def get_indices_of_k_smallest(arr, k):
        idx = np.argpartition(arr.ravel(), k)
        return tuple(np.array(np.unravel_index(idx, arr.shape))[:, range(min(k, 0), max(k, 0))])
    idx = get_indices_of_k_smallest(dist, percPixels)
    idx = np.array(idx,dtype=np.int32).T          
    area_lumen = np.zeros(mask.shape, dtype=np.int32)
    for i in range(idx.shape[0]):
        area_lumen[idx[i,0],idx[i,1]] = 1 
    return area_lumen

def stratified_gsm_features(img, mask, perimeter_lumen, percentages=[0.1, 0.25, 0.75, 0.9]):
    img = np.array(img,np.double)        
    mask = np.array(mask,np.int) 
    labels = ['stratified_GSM_' + perc for perc in percentages]
    
    features = []
    for perc in percentages:
        mask_perc = get_perc_ROI(mask, perimeter_lumen, perc)     
        img_ravel = img.ravel() 
        mask_perc_ravel = mask_perc.ravel()   
        roi = img_ravel[mask_perc_ravel.astype(bool)]   
        features.append(np.percentile(roi,50))
    features = np.array(features)
    
    return features, labels   