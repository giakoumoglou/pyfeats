# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Wed May 12 09:51:50 2021
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

def histogram(f, mask, bins=32): 
    ''' 
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    bins : int, optional
         Bins for histogram. The default is 32.


    Returns
    -------
    H : numpy ndarray
        Histogram of image f for 256 gray levels.
    labels : list
        Labels of features, which are the bins' number.
    '''
    
    if mask is None:
        mask = np.ones(f.shape)
        
    f  = f.astype(np.uint8)
    mask = mask.astype(np.uint8)
    level_min = 0
    level_max = 255

    f_ravel = f.ravel() 
    mask_ravel = mask.ravel() 
    roi = f_ravel[mask_ravel.astype(bool)] 
    H = np.histogram(roi, bins=bins, range=[level_min, level_max], density=True)[0]

    labels = ['Histogram_bin_'+str(b) for b in range(bins)] 
    return H, labels
        
def plot_histogram(f, mask, Ng=256, bins=32, name=''):
    if name != '':
        name = '('+name+')'
    f_ravel = f.ravel()
    mask_ravel = mask.ravel() 
    roi = f_ravel[mask_ravel.astype(bool)]
    plt.hist(roi, bins=bins, range=(0,Ng-1), density=True)     
    plt.title('Histogram: bins='+str(bins)+' Ng='+str(Ng)+' '+name)  
    plt.show()