# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Christos Loizou
@date: Sat May  8 09:42:10 2021
@reference: Wu, Texture Features for Classification
==============================================================================
Fractal Dimension Texture Analysis
==============================================================================
Inputs:
    - f:        image of dimensions N1 x N2
    - s:        degree to calculate Hurst coefficients
Outputs:
    - features: Hurst Coefficients
==============================================================================
"""
import numpy as np


def _least(id, k):
    '''
    Estimation of the curve slope using least squares regression, in log-log scale
    '''
    return np.polyfit(np.log10(np.arange(1,k+1)), np.log10(id), deg=1)[0]


def _intensity(f, mask, s):
    '''
    Intensity difference vector with step 
    '''
    n1, n2, cn1, cn2 = 0, 0, 0, 0
    N1, N2 = f.shape
    IDV = np.zeros((s), np.double)
        
    for k in range(1,s+1):
        for x1 in range(0,N1):
            for y1  in range(0,N2-k):
                if (mask[x1,y1] == 1) & (mask[x1,y1+k] == 1):
                    n1 += np.abs(f[x1,y1]-f[x1,y1+k])
                    cn1 += 1
    
        for x2 in range(0,N1-k):
            for y2 in range(0,N2):
                if (mask[x2,y2] == 1) & (mask[x2+k,y2] == 1):
                    n2 += np.abs(f[x2,y2]-f[x2+k,y2])
                    cn2 += 1
        IDV[k-1] = (n1+n2)/(cn1+cn2+1e-16)
          
    return IDV


def _resolution(x, mask, mr, mc):  
    '''
    Multiple resolution feature exctraction
    '''
    nr = (2 ** mr) - 1
    nc = (2 ** mc) - 1  
    nr_int = nr.astype('i')
    nc_int = nc.astype('i')
    res = np.zeros((nr_int,nc_int), np.double)
    res_mask = np.zeros((nr_int,nc_int), np.double)
    for i in range(0,nr_int):
        for j in range(0,nc_int):
            res[i,j]=(x[(2*i),(2*j)]+x[(2*i+1),(2*j)]+x[(2*i),(2*j+1)]+x[(2*i+1),(2*j+1)])/4
            res_mask[i,j]=(mask[(2*i),(2*j)]+mask[(2*i+1),(2*j)]+mask[(2*i),(2*j+1)]+mask[(2*i+1),(2*j+1)])/4
    res_mask[res_mask>1] = 1
    return res, res_mask
    

def fdta(f, mask, s=3):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    s : int, optional
        max resolution to calculate Hurst coefficients. The default is 3

    Returns
    -------
    h : numpy ndarray
        Hurst coefficients.
    labels : list
        Labels of h.
    '''
    
    if mask is None:
        mask = np.ones(f.shape)
        
    labels = ["FDTA_HurstCoeff"] * (s+1)
    labels = [label + "_" + str(i+1) for i,label in enumerate(labels)]
    f = np.array(f, np.double)
    N1, N2 = f.shape
    h = np.zeros((s+1), np.double)
    h[s] = 0
    ms = 3
    i = 0
    IDV = _intensity(f,mask,ms)
    h[i] = _least(IDV,ms)
    mr = np.log2(N1)
    mc = np.log2(N2)
    while (i < s):
        i = i + 1
        mr = mr - 1
        mc = mc - 1
        f, mask = _resolution(f,mask,mr,mc)
        IDV = _intensity(f,mask,ms)
        h[i] = _least(IDV,ms)
    return h, labels 
