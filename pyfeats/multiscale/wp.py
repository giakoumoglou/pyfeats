# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Thu May 13 12:14:15 2021
@reference: Tsiaparas, Comparison of Multiresolution Features for Texture Classification of Carotid Atherosclerosis From B-Mode Ultrasound
==============================================================================
"""

import pywt
import numpy as np

def wp_features(f, mask, wavelet='coif1', maxlevel=3): 
    ''' 
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    wavelet : str, optional
         Filter to be used. Check pywt for filter families. The default is 'cof1'
    maxlevel : int, optional
        Levels of decomposition. Default is 3.

    Returns
    -------
    features : numpy ndarray
        Mean and std of each detail image. Appromimation images are ignored.
    labels : list
        Labels of features.
    '''
    
    if mask is None:
        mask = np.ones(f.shape)
        
    # Step 1: Get Wavelet Decomposition in 3 levels 
    wp = pywt.WaveletPacket2D(data=f, wavelet=wavelet, mode='symmetric')
    wp_mask = pywt.WaveletPacket2D(data=mask, wavelet=wavelet, mode='symmetric')
    
    # Step 2: Check if we exceed max levels
    if maxlevel > wp.maxlevel:
        maxlevel = wp.maxlevel

    # Step 3: For each coeff array (64-1 sub-images), get mean and std
    features = np.zeros((len(wp.get_level(maxlevel))-1,2), np.double)
    paths = [node.path for node in wp.get_level(maxlevel, 'natural')]
    paths.remove('a' * maxlevel) # remove top left array
    labels = []
    for i in range(len(paths)):
        D_f = wp[paths[i]].data
        D_mask = wp_mask[paths[i]].data
        D_mask[D_mask != 0] = 1
        D = D_f.flatten()[D_mask.flatten().astype(bool)] # work on elements inside mask
        features[i][0], features[i][1] = abs(D).mean(), abs(D).std()
        labels.append('WP_' + str(wavelet) + '_' + str(paths[i]) + '_mean')
        labels.append('WP_' + str(wavelet) + '_' + str(paths[i]) + '_std')
    
    # Step 4: Return
    return features.flatten(), labels