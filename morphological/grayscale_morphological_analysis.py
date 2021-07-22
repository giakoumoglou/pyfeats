# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Mon May 10 12:42:30 2021
@reference: [33] Maragos, Pattern Spectrum and Multiscale Shape Representation
            [45] Maragos, Threshold Superposition in Morphological Image Analysis Systems
==============================================================================
B.3 Grayscale Mophological Analysis
==============================================================================
Inputs:
    - img:      image of dimensions N1 x N2
    - N:        scale
Outputs:
    - features: pdf, cdf [N x 1]
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology

def _opening_FP(f, g, n): # (f o ng), n=0,1,2... 
    out = f.copy()
    for i in range(n):
        out = morphology.erosion(out, g)
    for i in range(n):
        out = morphology.dilation(out,g)
    return out 

def _pattern_spectrum(f, g, n): # PS(f,g,n) = A[f o ng - f o (n+1)g] 
    ps = _opening_FP(f,g,n) - _opening_FP(f,g,(n+1))
    return ps.sum() 

def grayscale_morphology_features(f,N):
    f = f.astype(np.uint8)                # grayscale image
    kernel = np.ones((3,3), np.uint8)     # kerne: cross '+'
    kernel[0,0], kernel[2,2], kernel[0,2], kernel[2,0] = 0, 0, 0, 0 
    ps = np.zeros(N, np.double)           # pattern spectrum
    for n in range(N):
        ps[n] = _pattern_spectrum(f,kernel,n)
    pdf = ps / f.sum() 
    cdf = np.cumsum(pdf)  
    return pdf, cdf

def plot_pdf_cdf(pdf, cdf, name=''):
    if name != '':
        name = '('+name+')'
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle('Plaque Grayscale Morphological Features ' + str(name))
    ax1.plot(pdf)
    ax1.set_title('pdf')
    ax2.plot(cdf)
    ax2.set_title('cdf')