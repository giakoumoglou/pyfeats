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
"""

import numpy as np
from skimage.transform import radon
import matplotlib.pyplot as plt
import warnings
from .bispectrum import _bispectrum  

def _entropy(x):
    return -np.multiply(x, np.log(x+1e-16)).sum()

def hos_features(f, th=[135,140]):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    th : list, optional
        Angle to calculate Radon Transform. The default is [135,140].

    Returns
    -------
    features : numpy ndarray
        Entropy of bispectrum of radeon transform of image for each angle in theta.
    labels : list
        Labels of features.
    '''
    
    warnings.filterwarnings("ignore")
    f = f.astype(np.float32)
    th = np.array([th]).reshape(-1)
    N1, N2 = f.shape
    
    radon_transform = radon(f, theta=th)
    
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
