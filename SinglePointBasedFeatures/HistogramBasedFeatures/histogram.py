# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Wed May 12 09:51:50 2021
==============================================================================
C.1 Histogram
==============================================================================
Inputs:
    - f:        image of dimensions N1 x N2
    - mask:     int boolean image N1 x N2 with 1 if pixels belongs to ROI, 
                0 else
    - Ng:       number of gray levels
    - bins:     bins for histogram
Outputs:
    - features: histogram of f as vector e.g. [32 x 1]
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

def histogram(f, mask, bins=32):   
    level_min = 0
    level_max = 255
    f_ravel = f.ravel()
    mask_ravel = mask.ravel() 
    roi = f_ravel[mask_ravel.astype(bool)]
    H = np.histogram(roi, bins=bins, range=[level_min, level_max])[0]
    labels = ['Histogram_bin_'+str(b) for b in range(bins)] 
    return H, labels
        
def plot_histogram(f, mask, Ng=256, bins=32, name=''):
    if name != '':
        name = '('+name+')'
    f_ravel = f.ravel()
    mask_ravel = mask.ravel() 
    roi = f_ravel[mask_ravel.astype(bool)]
    plt.hist(roi, bins=bins, range=(0,Ng-1), density=True)     
    plt.title('Plaque Histogram: bins='+str(bins)+' Ng='+str(Ng)+' '+name)  
    plt.show()