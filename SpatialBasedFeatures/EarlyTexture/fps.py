# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: May  8 15:18:53 2021
@reference: [8] Weszka, A Comparative Study of Texture Measures for Terrain Classification
            [11] Wu, Texture Features for Classification
==============================================================================
A.9 Fourier Power Spectrum
==============================================================================
Inputs:
    - f:        image of dimensions N1 x N2
    - mask:     int boolean image N1 x N2 with 1 if pixels belongs to ROI, 
                0 else
Outputs:
    - features: 1)Radial Sum, 2)Angular Sum
==============================================================================
"""

import numpy as np
from ..utilities import _fft2

def fps(f, mask):
    
    # 1) Labels
    labels = ["FPS_RadialSum", "FPS_AngularSum"]
    
    # 2) Parameters
    f = np.array(f, np.double)
    mask = np.array(mask, np.double)
     
    # 3) Calculate Features
    area = mask.sum()
    F = _fft2(f, mask)
    F_real = np.real(F)
    F_imag = np.imag(F)
    F_real = np.multiply(F_real, mask)
    F_imag = np.multiply(F_imag, mask)
    features = np.zeros(2 ,np.double)
    features[0] = np.sqrt(sum(sum(np.multiply(F_real,F_real)))/area)
    features[1] = np.sqrt(sum(sum(np.multiply(F_imag,F_imag)))/area)
    return features, labels
 
