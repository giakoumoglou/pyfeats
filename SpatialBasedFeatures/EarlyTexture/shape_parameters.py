# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Sat May  8 12:14:26 2021
==============================================================================
A.10 Shape Parameters
==============================================================================
Inputs:
    - f:           image of dimensions N1 x N2
    - mask:        int boolean image N1 x N2 with 1 if pixels belongs to ROI, 
                   0 else
    - perimeter:   int boolean image N x M with 1 if pixels belongs to 
                   perimeter of ROI, 0 else
Outputs:
    - features:    1,2)X, Y coordinate max length, 3)area, 4)perimeter, 
                   5)perimeter2/area
==============================================================================
"""

import numpy as np

def shape_parameters(f, mask, perimeter, pixels_per_mm2=1, pad=4):
    
    # 1) Labels
    labels = ["SHAPE_XcoordMax", "SHAPE_YcoordMax", "SHAPE_area",
              "SHAPE_perimeter", "SHAPE_perimeter2perArea"]
    
    # 2) Parameters
    mask = np.array(mask, np.int32)
    perimeter = np.array(perimeter, np.int32)
    N1, N2 = f.shape
    
    # 3) Calculate Features
    features = np.zeros(5,np.double)
    features[0] = (N1 - pad) / pixels_per_mm2
    features[1] = (N2 - pad) / pixels_per_mm2
    features[2] = mask.sum() / (pixels_per_mm2 ** 2)
    features[3] = perimeter.sum() / pixels_per_mm2
    features[4] = (features[3] ** 2)/features[2]
    
    return features, labels  