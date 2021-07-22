# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Mon Thu May 13 13:14:35 2021
@reference: [41] Tsiaparas, Comparison of Multiresolution Features for Texture Classification of Carotid Atherosclerosis From B-Mode Ultrasound
==============================================================================
B.4 Gabor Transform (GT)
==============================================================================
Inputs:
    - f:        image of dimensions N1 x N2
    - mask:     int boolean image N1 x N2 with 1 if pixels belongs to ROI, 
                0 else
Outputs:
    - features:  mean and std [8 x 2 = 16] for fixed frquency and theta
==============================================================================
"""

import numpy as np
from skimage.filters import gabor_kernel
from scipy import signal
from ..utilities import _image_xor

def gt_features(f, mask):
    
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
            
    # Step 2: Get mask where convolution should be performed
    mask_c = _image_xor(mask)
    mask_convs = []
    for k in range(len(kernels)):
        oneskernel = np.ones(kernels[k].shape)
        temp = signal.convolve2d(mask_c, oneskernel,'same')
        temp = np.abs(np.sign(temp)-1)
        mask_convs.append(temp)
    
    # Step 3: Convolve image with each kernel and get mean and std
    features = np.zeros((len(kernels), 2), np.double)
    for k in range(len(kernels)):
            D = signal.convolve2d(f, kernels[k], 'same')
            D = np.multiply(D, mask_convs[k])
            D_ravel = D.ravel()
            mask_conv_ravel = mask_convs[k].ravel()
            roi = D_ravel[mask_conv_ravel==1]
            if roi.size == 0:
                features[k, 0], features[k, 1] = 0, 0
            else:
                features[k, 0], features[k, 1] = roi.mean(), roi.std()
                
    # step 4: Return
    return features.flatten(), labels


    
    
