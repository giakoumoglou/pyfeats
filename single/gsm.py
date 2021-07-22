# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Thu May  6 21:59:57 2021
==============================================================================
A.1 Gray Scale Median
==============================================================================
Inputs:
    - img:      image
    - mask:     2D array with 1 inside ROI [int32]
Outputs:
    - features: Gray Scale Median (GSM)
==============================================================================
"""

import numpy as np

def gsm_feature(img, mask):
    img = np.array(img,np.double)        
    mask = np.array(mask,np.int)   
    Ng = 256        
    img_ravel = img.ravel() 
    mask_ravel = mask.ravel() 
    roi = img_ravel[mask_ravel.astype(bool)]  
    features = np.percentile(roi,50) 
    return features, ['GSM']   