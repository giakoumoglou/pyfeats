# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Sat May  8 12:14:26 2021
==============================================================================
"""

import numpy as np

def shape_parameters(f, mask, perimeter, pixels_per_mm2=1):
    ''' 
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    perimeter : numpy ndarray
         Image N1 x N2 with 1 if pixels belongs to perimeter of ROI, 0 else.
    pixels_per_mm2 : int, optional
        Density of image f. The default is 1.

    Returns
    -------
    features : numpy ndarray
        1,2)X, Y coordinate max length, 3)area, 4)perimeter, 5)perimeter2/area.
    labels : list
        Labels of features.
    '''

    if mask is None:
        mask = np.ones(f.shape)
        
    # 1) Labels
    labels = ["SHAPE_XcoordMax", "SHAPE_YcoordMax", "SHAPE_area",
              "SHAPE_perimeter", "SHAPE_perimeter2perArea"]
    
    # 2) Parameters
    mask = np.array(mask, np.int32)
    perimeter = np.array(perimeter, np.int32)
    N1, N2 = mask.shape
    
    # 3) Find X max coordinate
    start, end = 0, N1
    for x in range(N1):
        if (sum(mask[x,:])>0) & (start==None):
            start = x
        if (sum(mask[x,:])==0) & (start != None):
            end = x-1
            break
    Xcoord_max = end - start + 1
    
    # 4) Find Y max coordinate
    start, end = 0, N2
    for x in range(N2):
        if (sum(mask[:,x])>0) & (start==None):
            start = x
        if (sum(mask[:,x])==0) & (start != None):
            end = x-1
            break
    Ycoord_max = end - start + 1
            
    # 5) Calculate Features
    features = np.zeros(5,np.double)
    features[0] = Xcoord_max / pixels_per_mm2
    features[1] = Ycoord_max / pixels_per_mm2
    features[2] = mask.sum() / (pixels_per_mm2 ** 2)
    features[3] = perimeter.sum() / pixels_per_mm2
    features[4] = (features[3] ** 2)/features[2]
    
    return features, labels  
