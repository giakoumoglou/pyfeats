# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Thu May 13 09:50:26 2021
@reference: Ojala, A Comparative Study of Texture Measures with Classification on Feature Distributions
            Ojala, Gray Scale and Roation Invariaant Texture Classification with Local Binary Patterns
==============================================================================
"""

import numpy as np
from skimage import feature

def _energy(x):
    return np.multiply(x,x).sum()

def _entropy(x):
    return -np.multiply(x, np.log(x+1e-16)).sum()

def lbp_features(f, mask, P=[8,16,24], R=[1,2,3]):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    P : list, optional
        Number of points in neighborhood. The default is [8,16,24].
    R : list, optional
        Radius/Radii. The default is [1,2,3].

    Returns
    -------
    features : numpy ndarray
        Energy and entropy of LBP image (2 x 1).
    labels : list
        Labels of features.
    '''
    
    if mask is None:
        mask = np.ones(f.shape)
        
    P = np.array(P)
    R = np.array(R)
    n = P.shape[0]
    mask_ravel = mask.ravel() 
    features = []
    labels = []
    
    for i in range(n):
        lbp = feature.local_binary_pattern(f, P[i], R[i], 'uniform')
        lbp_ravel = lbp.ravel() 
        roi = lbp_ravel[mask_ravel.astype(bool)] 
        feats = np.zeros(2, np.double)
        feats[0] = _energy(roi) / roi.sum()
        feats[1] = _entropy(roi) / roi.sum()
        features.append(feats)
        labels.append('LBP_R_'+str(R[i])+'_P_'+str(P[i])+'_energy')
        labels.append('LBP_R_'+str(R[i])+'_P_'+str(P[i])+'_entropy')
        
    features = np.array(features, np.double).ravel()
    
    return features, labels

    