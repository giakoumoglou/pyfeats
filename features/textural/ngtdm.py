 # -*- coding: utf-8 -*-
"""
==============================================================================
@author: Christos Loizou
@author: Nikolaos Giakoumoglou
@date: Fri May  7 13:53:51 2021
@reference: Amadasun, Texural Features Corresponding to Textural Properties
==============================================================================
"""
import numpy as np
from scipy import signal
from ..utilities import _image_xor

def ngtdm(f, mask, d, Ng=256):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else.
    d : int, optional
        Distance for NGTDM. Default is 1.
    Ng : int, optional
        Image number of gray values. The default is 256.

    Returns
    -------
    S : numpy ndarray
    N : numpy ndarray
    R : numpy ndarray
    '''
    
    f = f.astype(np.double)
    N1, N2 = f.shape
    oneskernel = np.ones((2*d+1,2*d+1))
    kernel = oneskernel.copy()
    kernel[d,d] = 0
    W = (2*d + 1)**2 
    
    # Get complementary mask     
    mask_c = _image_xor(mask)
    
    # Find which pixels are inside mask for convolution
    conv_mask = signal.convolve2d(mask_c,oneskernel,'same')
    conv_mask = abs(np.sign(conv_mask)-1)
        
    # Calculate abs diff between actual and neighborhood
    A = signal.convolve2d(f,kernel,'same') / (W-1)
    diff = abs(f-A)
         
    # Construct NGTDM matrix
    S = np.zeros(Ng,np.double)
    N = np.zeros(Ng,np.double)
    for x in range(d,(N1-d)):
        for y in range(d,(N2-d)):
        	if conv_mask[x,y] > 0:
        		index = f[x,y].astype('i')
        		S[index] = S[index] + diff[x,y]
        		N[index] += 1
            
    R = sum(N)
    
    return S, N, R

def ngtdm_features(f, mask, d=1):
    '''  
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    d : int, optional
        Distance for NGTDM. Default is 1.

    Returns
    -------
    features : numpy ndarray
        1)Coarseness, 2)Contrast, 3)Busyness, 4)Complexity, 5)Strength.
    labels : list
        Labels of features.
    '''
    
    if mask is None:
        mask = np.ones(f.shape)
        
    # 1) Labels
    labels = ["NGTDM_Coarseness","NGTDM_Contrast","NGTDM_Busyness",
              "NGTDM_Complexity","NGTDM_Strngth"]
    
    # 2) Parameters
    f  = f.astype(np.uint8)
    mask = mask.astype(np.uint8)
    Ng = 256
    
    # 3) Calculate NGTDM
    S, N, R = ngtdm(f, mask, d, Ng)
        
    # 4) Calculate Features
    features = np.zeros(5,np.double) 
    Ni, Nj = np.meshgrid(N,N)
    Si, Sj = np.meshgrid(S,S)
    i, j = np.meshgrid(np.arange(Ng),np.arange(Ng))
    ilessjsq = ((i-j)**2).astype(np.double)   
    Ni = np.multiply(Ni,abs(np.sign(Nj)))
    Nj = np.multiply(Nj,abs(np.sign(Ni)))     
    features[0] = R*R / sum(np.multiply(N,S))
    features[1] = sum(S)*sum(sum(np.multiply(np.multiply(Ni,Nj),ilessjsq)))/R**3/Ng/(Ng-1)
    temp = np.multiply(i,Ni) - np.multiply(j,Nj)
    features[2] = sum(np.multiply(N,S)) / sum(sum(abs(temp))) / R
    temp = np.multiply(Ni,Si) + np.multiply(Nj,Sj)
    temp2 = np.multiply(abs(i-j),temp)
    temp3 = np.divide(temp2,Ni+Nj+1e-16)
    features[3] = sum(sum(temp3)) / R
    features[4] = sum(sum(np.multiply(Ni+Nj,ilessjsq))) / (sum(S)+1e-16)
        
    return features, labels