# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Mon May 10 12:42:30 2021
@reference: [12] Acharya, AtheromaticTM Symptomatic vs. Asymptomatic Classification
            [13] Acharya, Carotid Ultrasound Symptomatology using Atherosclerotic Plaque
            [41] Tsiaparas, Comparison of Multiresolution Features for Texture Classification of Carotid Atherosclerosis From B-Mode Ultrasound
==============================================================================
C.5 Discrete Wavelet Transform (DWT)
==============================================================================
Inputs:
    - f:         image of dimensions N1 x N2
    - wavelet:   family of filter for DWT (default='bior3.3')
    - level:     level for DWT decomposition (default=3)
Outputs:
    - features:  mean and std of each cD, cH, cV [9 x 2 = 18]
==============================================================================
"""

import pywt
import numpy as np
from ..utilities import _mean_std

def dwt_features(f, wavelet='bior3.3',levels=3):
    
    # Step 1: Get DWT Decomposition for 3 levels
    coeffs = pywt.wavedec2(f, wavelet=wavelet, level=levels)
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    
    # Step 2: For each coeff array (10-1=9 sub-images), get mean and std
    labels = []
    features = np.zeros((3*levels,2),np.double)
    i = 0
    for level in range(1,levels+1):
        for name in ['da','dd','ad']:
            D =  coeff_arr[coeff_slices[level][name]]
            mi, sigma = _mean_std(D)
            features[i][0], features[i][1] = mi, sigma
            i += 1
            labels.append('DWT_' + str(wavelet) + '_level_' + str(level) + 
                          '_' + str(name) + '_mean')
            labels.append('DWT_' + str(wavelet) + '_level_' + str(level) + 
                          '_' + str(name) + '_std')
            
    # Step 3: Return
    return features.flatten(), labels