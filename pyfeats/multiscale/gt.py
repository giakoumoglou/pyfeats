# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Mon Thu May 13 13:14:35 2021
@reference: Tsiaparas, Comparison of Multiresolution Features for Texture Classification of Carotid Atherosclerosis From B-Mode Ultrasound
==============================================================================
"""

import numpy as np
from skimage.filters import gabor_kernel
from scipy import signal
from ..utilities import _image_xor

def gt_features(f, mask, deg=4, freq=[0.05, 0.4]):
    ''' 
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    deg: int, optinal
        Quantized degrees. The default is 4 (0, 45, 90, 135 degrees)
    freq: list, optional
        frequency of the gabor kernel. The default is [0.05, 0.4]

    Returns
    -------
    features : numpy ndarray
        Mean and std for the resulted image: (f o gabor_filter)(x,y)
        
    labels : list
        Labels of features.
    '''    
    
    if mask is None:
        mask = np.ones(f.shape)
        
    # Step 1: Initialize kernels
    kernels = []
    labels = []
    for theta in range(deg): # e.g. th = 0, 45, 90, 135 degrees
        theta = theta / deg * np.pi
        for frequency in freq: # e.g. f = 0.05 and 0.4
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


    
    
