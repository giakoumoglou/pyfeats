# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Thu May 13 12:14:15 2021
@reference: [41] Tsiaparas, Comparison of Multiresolution Features for Texture Classification of Carotid Atherosclerosis From B-Mode Ultrasound
==============================================================================
C.7 Wavelet Packets (WP)
==============================================================================
Inputs:
    - f:         image of dimensions N1 x N2
    - wavelet:   family of filter for DWT (default='coif1')
    - maxlevel:  levels for wavelet decomposition (default=3)
Outputs:
    - features: mean and std [63 x 2 = 126]
==============================================================================
"""

import pywt
import numpy as np
from ..utilities import _mean_std

def wp_features(f, wavelet='coif1', maxlevel=3):
    
    # Step 1: Get Wavelet Decomposition in 3 levels 
    wp = pywt.WaveletPacket2D(data=f, wavelet=wavelet, mode='symmetric')
    
    # Step 2: Check if we exceed max levels
    if maxlevel > wp.maxlevel:
        maxlevel = wp.maxlevel

    # Step 3: For each coeff array (64-1 sub-images), get mean and std
    features = np.zeros((len(wp.get_level(maxlevel))-1,2), np.double)
    paths = [node.path for node in wp.get_level(maxlevel, 'natural')]
    paths.remove('a' * maxlevel) # remove top left array
    labels = []
    for i in range(len(paths)):
        D = wp[paths[i]].data
        mi, sigma = _mean_std(D)
        features[i][0] = mi
        features[i][1] = sigma
        labels.append('WP_' + str(wavelet) + '_' + str(paths[i]) + '_mean')
        labels.append('WP_' + str(wavelet) + '_' + str(paths[i]) + '_std')
    
    # Step 4: Return
    return features.flatten(), labels

    