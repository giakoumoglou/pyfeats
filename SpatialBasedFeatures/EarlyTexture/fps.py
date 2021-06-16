# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: May  8 15:18:53 2021
@reference: [8] Weszka, A Comparative Study of Texture Measures for Terrain Classification
            [11] Wu, Texture Features for Classification
==============================================================================
B.1 Fourier Power Spectrum
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

def _fft2(x,msk):
    r,c = x.shape
    f1 = np.zeros((r,c), np.complex64)
    
    for i in range(1,c+1):
      sp = 0
      ep = 0
      j = 1
      while (j < r):
          while (msk[j-1,i-1]==1) & (j < r):
              if (sp == 0):
                  sp = j
              j += 1
          if (sp > 0) & (ep == 0):
              if (j < r):
                  ep = j-1
              else:
                  ep = j      
              f1[(sp-1):(ep),i-1] = np.fft.fft(x[(sp-1):(ep),i-1])
              sp = 0
              ep = 0
          while (msk[j-1,i-1] == 0) & (j < r):
              j += 1

    f1 = f1.T
    msk = msk.T
    
    for i in range(1,r+1):
        sp = 0
        ep = 0
        j = 1
        while (j < c):
          while (msk[j-1,i-1] == 1) & (j < c):
              if (sp == 0):
                  sp = j
              j += 1
          if (sp > 0) & (ep == 0):
              if (j < r):
                  ep = j-1
              else:
                  ep = j     
              f1[(sp-1):(ep),i-1] = np.fft.fft(f1[(sp-1):(ep),i-1])
              sp = 0
              ep = 0
          while (msk[j-1,i-1] == 0) & (j < c):
              j += 1

    f1 = f1.T
    msk = msk.T
    return f1

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
 
