# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Mon May 10 12:42:30 2021
@reference: Acharya, AtheromaticTM Symptomatic vs. Asymptomatic Classification
            Acharya, Carotid Ultrasound Symptomatology using Atherosclerotic Plaque
            Tsiaparas, Comparison of Multiresolution Features for Texture Classification of Carotid Atherosclerosis From B-Mode Ultrasound
==============================================================================
Discrete Wavelet Transform (DWT)
==============================================================================
Inputs:
    - f:         image of dimensions N1 x N2
    - mask:      int boolean image N1 x N2 with 1 if pixels belongs to ROI, 
                 0 else
    - wavelet:   family of filter for DWT (default='bior3.3')
    - level:     level for DWT decomposition (default=3)
Outputs:
    - features:  mean and std of each cD, cH, cV [9 x 2 = 18]
==============================================================================
"""

import pywt
import numpy as np
from ..utilities import _pad_image_power_2

def dwt_features(f, mask, wavelet='bior3.3',levels=3):
    
    # Step 1: Pad Image
    f = _pad_image_power_2(f)       # pad to the next power of 2 in each dimension
    mask = _pad_image_power_2(mask) # pad to the next power of 2 in each dimension
        
    # Step 2: Get DWT Decomposition for 3 levels
    coeffs = pywt.wavedec2(f, wavelet=wavelet, level=levels)
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
    
    # Step 3: Get DWT Decomposition for 3 levels for mask
    coeffs_mask = pywt.wavedec2(mask, wavelet=wavelet, level=levels)
    coeff_arr_mask, coeff_slices_mask = pywt.coeffs_to_array(coeffs_mask)
    coeff_arr_mask[coeff_arr_mask!=0] = 1
            
    # Step 4: For each coeff array (10-1=9 sub-images), get mean and std
    labels = []
    features = np.zeros((3*levels,2),np.double)
    i = 0
    for level in range(1,levels+1):
        for name in ['da','dd','ad']:
            D_f =  coeff_arr[coeff_slices[level][name]]
            D_mask = coeff_arr_mask[coeff_slices[level][name]]
            D = D_f.flatten()[D_mask.flatten().astype(bool)] # work on elements inside mask
            features[i][0], features[i][1] = abs(D).mean(), abs(D).std()
            i += 1
            labels.append('DWT_' + str(wavelet) + '_level_' + str(level) + 
                          '_' + str(name) + '_mean')
            labels.append('DWT_' + str(wavelet) + '_level_' + str(level) + 
                          '_' + str(name) + '_std')
            
    # Step 5: Return
    return features.flatten(), labels