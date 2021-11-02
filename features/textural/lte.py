# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Fri May  7 19:28:12 2021
@reference: Wu, Texture Features for Classification
            Law, Rapid Texture Identification
            Haralick, Computer and Robot Vision Vol. 1
==============================================================================
"""

import numpy as np
from scipy import signal
from ..utilities import _image_xor

def lte_measures(f, mask, l=7):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    l : int, optional
        Law's mask size. The default is 7.

    Returns
    -------
    features : numpy ndarray
        1)texture energy from LL kernel, 2) texture energy from EE 
        kernel, 3)texture energy from SS kernel, 4)average texture 
        energy from LE and EL kernels, 5)average texture energy from 
        ES and SE kernels, 6)average texture energy from LS and SL 
        kernels.
    labels : list
        Labels of features.
    '''

    if mask is None:
        mask = np.ones(f.shape)
        
    # 1) Labels
    labels = ["LTE_LL","LTE_EE","LTE_SS","LTE_LE","LTE_ES","LTE_LS"]
    labels = [label+'_'+str(l) for label in labels]
    
    # 2) Parameters
    f = np.array(f, np.double)
    mask = np.array(mask, np.double) 
    kernels = np.zeros((l,l,9), np.double)
    
    # 3) From 3 kernels [L, E, S], get 9 [LL, LE, LS, EL, EE, ES, SL, SE, SS]
    if l==3:
        L = np.array([ 1,  2,  1], np.double)
        E = np.array([-1,  0,  1], np.double)
        S = np.array([-1,  2, -1], np.double)
    elif l==5:
        L = np.array([ 1,  4,  6,  4,  1], np.double)
        E = np.array([-1, -2,  0,  2,  1], np.double)
        S = np.array([-1,  0,  2,  0, -1], np.double)
    else:
        L = np.array([ 1,  6,  15,  20,  15,  6,  1], np.double)
        E = np.array([-1, -4,  -5,   0,   5,  4,  1], np.double)
        S = np.array([-1, -2,   1,   4,   1, -2, -1], np.double)
    oneskernel = np.ones((l,l), np.double)    
    kernels = np.zeros((l,l,9), np.double)
    kernels[:,:,0] = np.multiply(L.reshape(-1,1),L) # LL kernel
    kernels[:,:,1] = np.multiply(L.reshape(-1,1),E) # LE kernel
    kernels[:,:,2] = np.multiply(L.reshape(-1,1),S) # LS kernel
    kernels[:,:,3] = np.multiply(E.reshape(-1,1),L) # EL kernel
    kernels[:,:,4] = np.multiply(E.reshape(-1,1),E) # EE kernel
    kernels[:,:,5] = np.multiply(E.reshape(-1,1),S) # ES kernel
    kernels[:,:,6] = np.multiply(S.reshape(-1,1),L) # SL kernel
    kernels[:,:,7] = np.multiply(S.reshape(-1,1),E) # SE kernel
    kernels[:,:,8] = np.multiply(S.reshape(-1,1),S) # SS kernel
    
    # 4) Get mask where convolution should be performed
    mask_c = _image_xor(mask)
    mask_conv = signal.convolve2d(mask_c, oneskernel,'valid')
    mask_conv = np.abs(np.sign(mask_conv)-1)
        
    # 5) Calculate energy of each convolved image with each kernel: total 9
    energy = np.zeros(9,np.double)   
    area = sum(sum(mask_conv))          
    for i in range(9):
        f_conv = signal.convolve2d(f, kernels[:,:,i], mode='valid')
        f_conv = np.multiply(f_conv,mask_conv)     
        f_conv_mean = sum(sum(f_conv)) / area
        energy[i] = np.sqrt(sum(sum(np.multiply((f_conv-f_conv_mean)**2,mask_conv)))/area)
           
    # 6) Calculate features
    features = np.zeros(6,np.double) 
    features[0] = energy[0]
    features[1] = energy[4]
    features[2] = energy[8]
    features[3] = (energy[1]+energy[3])/2
    features[4] = (energy[5]+energy[7])/2
    features[5] = (energy[2]+energy[6])/2
        
    return features, labels


