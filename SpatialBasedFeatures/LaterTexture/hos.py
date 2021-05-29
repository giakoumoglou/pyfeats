# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Wed May 12 17:37:03 2021
@reference: [35] Chua, Automatic indentification of epilepsy by hos and power spectrum parameters using eeg signals
            [36] Chua, Application of Higher Order Spectra to Identify Epileptic eeg
            [37] Acharya, Automatic identification of epileptic eeg singal susing nonlinear parameters
            [38] Acharya, Application of higher order spectra for the identification of diabetes retinopathy stages
==============================================================================            
C.4 Higher Order Spectra on Radeon Transform
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

import numpy as np
from skimage.transform import radon
import matplotlib.pyplot as plt
import warnings
from ..utilities import _entropy
from .bispectrum import _bispectrum  

def hos_features(f, th=[135,140]):
    
    warnings.filterwarnings("ignore")
    f = f.astype(np.double)
    th = np.array([th]).reshape(-1)
    N1, N2 = f.shape
    
    radon_transform = radon(f.astype(np.uint8), theta=th)
    
    entropy = []
    labels = ['HOS_'+str(t)+'_degrees' for t in th]
    for i in range(th.shape[0]):
        B, _ = _bispectrum(radon_transform[:,i])
        p = abs(B) / (sum(sum(abs(B))))
        e = _entropy(p)
        entropy.append(e)
        
    warnings.filterwarnings("default")
    return np.array(entropy).reshape(-1), labels

def plot_sinogram(f, name=''):
    if name != '':
        name = '('+name+')'
    warnings.filterwarnings("ignore")
    th = np.linspace(0., 180., max(f.shape), endpoint=False)
    sinogram = radon(f, theta=th)
    dx, dy = 0.5 * 180.0 / max(f.shape), 0.5 / sinogram.shape[0]
    plt.imshow(sinogram, cmap=plt.cm.Greys_r,
           extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
           aspect='auto')
    plt.title('Sinogram '+name)
    warnings.filterwarnings("default")