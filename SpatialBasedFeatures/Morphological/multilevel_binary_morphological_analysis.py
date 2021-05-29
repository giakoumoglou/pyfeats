# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Tue May 11 13:20:14 2021
@reference: [33] Maragos, Pattern Spectrum and Multiscale Shape Representation
            [45] Maragos, Threshold Superposition in Morphological Image Analysis Systems
==============================================================================
Multilevel Binary Morphological Analysis
See the reference for more details
==============================================================================
Inputs:
    - img:      image of dimensions N1 x N2
    - mask:     int boolean image with 1 if pixels belongs to ROI, 0 else
    - N:        scale
Outputs:
    - features: pdf, cdf for each binary image L, M, H [N x 3]
==============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology

def _get_binary_images(img, mask): 
    L = np.zeros(img.shape,np.uint8)
    M = np.zeros(img.shape,np.uint8)
    H = np.zeros(img.shape,np.uint8)   
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if mask[i,j] == 1:
                if img[i,j] < 25:
                    L[i,j] = 255
                elif img[i,j] > 50:
                    H[i,j] = 255
                else:
                    M[i,j] = 255
    return L, M, H

def _opening_SP(X,B,n): # (X o nB), n=0,1,2...
    out = X.copy()
    for i in range(n):
        out = morphology.binary_erosion(out, B)
    for i in range(n):
        out = morphology.binary_dilation(out,B)
    return out.astype(np.uint8)

def _pattern_spectrum(X,B,n): # PS(X,B,n) = A[X o nB - X o (n+1)B] 
    ps = _opening_SP(X,B,n) - _opening_SP(X,B,n+1)
    return np.count_nonzero(ps)

def _multilevel_binary_morphological_analysis(X, B, N):        
    ps = np.zeros(N,np.double)
    for n in range(N):
        ps[n] = _pattern_spectrum(X,B,n)
    pdf = ps / np.count_nonzero(X)
    cdf = np.cumsum(pdf)
    return pdf, cdf

def multilevel_binary_morphology_features(img, mask, N):
    img = img.astype(np.uint8) # grayscale image
    mask = mask.astype(np.uint8)
    L, M, H = _get_binary_images(img, mask)
    kernel = np.ones((3,3), np.uint8) # kernel/structuring element
    kernel[0,0], kernel[2,2], kernel[0,2], kernel[2,0] = 0, 0, 0, 0 # cross '+'
    pdfs, cdfs = np.zeros((N,3),np.double), np.zeros((N,3),np.double)
    for i, a in enumerate([L, M, H]):
        pdfs[:,i], cdfs[:,i] = _multilevel_binary_morphological_analysis(a,kernel,N)    
    pdf_L, pdf_M, pdf_H = np.array_split(pdfs, 3, axis=1)
    cdf_L, cdf_M, cdf_H = np.array_split(cdfs, 3, axis=1)
    return pdf_L.flatten(), pdf_M.flatten(), pdf_H.flatten(), cdf_L.flatten(), cdf_M.flatten(), cdf_H.flatten()

def plot_pdfs_cdfs(pdf_L, pdf_M, pdf_H, cdf_L, cdf_M, cdf_H, name=''):
    
    if name != '':
        name = '('+name+')'
        
    fig, axs = plt.subplots(3,2)
    fig.suptitle('Plaque Multilevel Binary Morphological Features ' + str(name))
    
    axs[0,0].plot(pdf_L)
    axs[0,1].plot(cdf_L)
    axs[0,0].set_title('pdf low image')
    axs[0,1].set_title('cdf low image')
    
    axs[1,0].plot(pdf_M)
    axs[1,1].plot(cdf_M)
    axs[1,0].set_title('pdf mid image')
    axs[1,1].set_title('cdf mid image')
    
    axs[2,0].plot(pdf_H)
    axs[2,1].plot(cdf_H)
    axs[2,0].set_title('pdf high image')
    axs[2,1].set_title('cdf high image')