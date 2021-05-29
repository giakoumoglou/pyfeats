# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Mon Thu May 13 13:14:35 2021
@reference: [41] Tsiaparas, Comparison of Multiresolution Features for Texture Classification of Carotid Atherosclerosis From B-Mode Ultrasound
==============================================================================
C.8 Gabor Transform (GT)
==============================================================================
Inputs:
    - f:        image of dimensions N1 x N2
Outputs:
    - features:  mean and std [8 x 2 = 16] for fixed frquency and theta
==============================================================================
"""

import numpy as np
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
from ..utilities import _mean_std

def gt_features(f):
    
    # Step 1: Initialize kernels
    kernels = []
    labels = []
    for theta in range(4): # th = 0, 45, 90, 135 degrees
        theta = theta / 4. * np.pi
        for frequency in (0.05, 0.4): # f = 0.05 and 0.4
            kernel = np.real(gabor_kernel(frequency, theta=theta))
            kernels.append(kernel)
            labels.append('GT_th_' + str(theta*4/np.pi) + '_freq_' + 
                          str(frequency) + '_mean')
            labels.append('GT_th_' + str(theta*4/np.pi) + '_freq_' + 
                          str(frequency) + '_std')
            
    # Step 2: Convolve image with each kernel and get mean and std
    features = np.zeros((len(kernels), 2), np.double)
    for k, kernel in enumerate(kernels):
            D = ndi.convolve(f, kernel, mode='wrap')
            mi, sigma = _mean_std(D)
            features[k, 0], features[k, 1] = mi, sigma
    
    # step 3: Return
    return features.flatten(), labels
    
    
