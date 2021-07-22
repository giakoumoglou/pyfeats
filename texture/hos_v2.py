# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Wed May 12 17:37:03 2021
@reference: Chua, Automatic indentification of epilepsy by hos and power spectrum parameters using eeg signals
            Chua, Application of Higher Order Spectra to Identify Epileptic eeg
            Acharya, Automatic identification of epileptic eeg singal susing nonlinear parameters
            Acharya, Application of higher order spectra for the identification of diabetes retinopathy stages
==============================================================================            
Higher Order Spectra on Radeon Transform
==============================================================================
1. Image 2D I(x,y)
2. Radon Transform (theta) -> output: projection 1D
3. Bispectrum of projection -> output: 2D array f1 x f2 (=128)
4. Features: entropy of 2D array f1 x f2
==============================================================================
Inputs:
    - f:        image of dimensions N1 x N2
    - th:       theta to calculate radeon transform (135,140 used in [12])
Outputs:
    - features: entropy of bispectrum of radeon transform of image for each 
                angle in theta
==============================================================================
"""

from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import math

from .bispectrum import _bispectrum

def _pad_image_2(f):
    N1, N2 = f.shape
    N1_deficit = N2+2
    N2_deficit = N1+2
    f2 = np.pad(f, ((math.floor(N1_deficit/2),N1_deficit-math.floor(N1_deficit/2)), 
                    (math.floor(N2_deficit/2),N2_deficit-math.floor(N2_deficit/2))), 
                mode='constant')
    return f2

def discrete_radon_transform(f, theta, remove_zeros=False):
    f2 = _pad_image_2(f) # so i can rotate without loosing information
    res = np.zeros((len(f2[0]), len(theta)), dtype='float64')
    for i,th in enumerate(theta):
        rotation = ndimage.rotate(f2, th, reshape=False).astype('float64')
        res[:,i] = sum(rotation).reshape(-1)
    if remove_zeros == True:
        res = res[~np.all(abs(res) < 1e-16, axis=1)]
    return res

def _entropy(x):
    return -np.multiply(x, np.log(x+1e-16)).sum()


def hos_features(f, th=[135,140]):
    
    f = f.astype(np.float32)   
    radon_transform = discrete_radon_transform(f,th, remove_zeros=True)
    
    labels = ['HOS_'+str(th)+'_degrees' for th in th]
    
    entropy = []
    for i in range(len(th)):
        B, _ = _bispectrum(radon_transform[:,i])
        p = abs(B) / abs(B).sum()
        e = _entropy(p)
        entropy.append(e)
    
    return np.array(entropy).reshape(-1), labels

def plot_sinogram(f, name=''):
    if name != '':
        name = '('+name+')'
    theta = [i for i in range(180)]
    sinogram = discrete_radon_transform(f, theta, True)
    
    dx, dy = 0.5 * 180.0 / max(f.shape), 0.5 / sinogram.shape[0]
    plt.imshow(sinogram, cmap='gray',
           extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
           aspect='auto')
    plt.title('Sinogram '+name)
    plt.show()