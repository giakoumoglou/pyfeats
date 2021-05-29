# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Christos Loizou
@date: Sat May  8 09:42:10 2021
@reference: [11] Wu, Texture Features for Classification
==============================================================================
A.7 Fractal Dimension Texture Analysis
==============================================================================
Inputs:
    - f:        image of dimensions N1 x N2
    - s:        degree to calculate Hurst coefficients
Outputs:
    - features: Hurst Coefficients
==============================================================================
"""
import numpy as np

# Estimation of the curve slope using least squares regression, in log-log scale
def least(id, r):
    return np.polyfit(np.log10(np.arange(1,r+1)), np.log10(id), deg=1)[0]

# Intensity difference vector with step 
def intensity(x, s):
    n1, n2, cn1, cn2 = 0, 0, 0, 0
    r, c = x.shape
    y = np.zeros((s), np.double)
        
    for k in range(1,s+1):
        for r1 in range(0,r):
            for c1  in range(0,c-k):
                if (x[r1,c1] < 255) & (x[r1,c1+k] < 255):
                    n1 += np.abs(x[r1,c1]-x[r1,c1+k])
                    cn1 += 1
    
        for r2  in range(0,r-k):
            for c2 in range(0,c):
                if (x[r2,c2] < 255) & (x[r2+k,c2] < 255):
                    n2 += np.abs(x[r2,c2]-x[r2+k,c2])
                    cn2 += 1
        y[k-1] = (n1+n2)/(cn1+cn2)
          
    return y

# Multiple resolution feature exctraction
def resolution(x, mr, mc):
    r, c = x.shape   
    nr = (2 ** mr) - 1
    nc = (2 ** mc) - 1  
    nr_int = nr.astype('i')
    nc_int = nc.astype('i')
    res = np.zeros((nr_int,nc_int), np.double)
    for i in range(0,nr_int):
        for j in range(0,nc_int):
            res[i,j]=(x[(2*i),(2*j)]+x[(2*i+1),(2*j)]+x[(2*i),(2*j+1)]+x[(2*i+1),(2*j+1)])/4
    return res
    
# Fractal Dimension Texture Analysis
def fdta(f, s=3):
    labels = ["FDTA_HurstCoeff"] * (s+1)
    labels = [label + "_" + str(i+1) for i,label in enumerate(labels)]
    x = f.copy()
    x = np.array(x, np.double)
    r, c = x.shape
    h = np.zeros((s+1), np.double)
    h[s] = 0
    ms = 3
    i = 0
    y = intensity(x,ms)
    h[i] = least(y,ms)
    mr = np.log2(r)
    mc = np.log2(c)
    while (i < s):
        i = i + 1
        mr = mr - 1
        mc = mc - 1
        x = resolution(x,mr,mc)
        y = intensity(x,ms)
        h[i] = least(y,ms)
    return h, labels 
