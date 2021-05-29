# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Kyriacos Constantinou
@author: Ioannis Constantinou,
@author: Marios S. Pattichis
@author: Constantinos S. Pattichis
@reference: [43] Murray, An AM-FM model for Motion Estimation in Atherosclerotic Plaque Videos
            [44] Murray, Multiscale AMFM Demodulation and Image Reconstruction methods with Improved Accuracy 
            [57] Pattichis, Medical Image Analysis Using AM-FM Models and Methods
==============================================================================
C.9 Amplitude Modulation - Frequency Modulation (AM-FM) using Gabor Filerbank
==============================================================================
Inputs:
    - f:        image of dimensions N1 x N2
    - mask:     int boolean image N1 x N2 with 1 if pixels belongs to ROI, 
                0 else
Outputs:
    - features: histogram of IA, IP, IFx, IFy with 32 bins fixed, a feature
                vector of 128
==============================================================================
"""

import numpy as np
from  scipy import signal
from ..utilities import gaborKernel2D,gaussianFunction
import warnings
 
def _filterbank():
    lamda0 = 2
    orientations = 8
    scales = 5
    gamma = 1
    phase = 0
    bandwidth = 1
    overlapIndex = 0.5
    offset = 0
    theta = np.arange(offset,np.pi - np.pi/orientations + offset + 
                             np.pi/orientations, (np.pi/orientations))
    lamda = lamda0
    filters = []
    
    for sc_index in range(scales, 0, -1):
        lamda0 = lamda
        for th in range(theta.shape[0]):
            result, sig = gaborKernel2D(theta[th], lamda, gamma, bandwidth, phase, 1/overlapIndex);
            filters .append(result)
        lamda = lamda0 * 2**bandwidth
    # Add DC Filter
    f1      = 2*np.pi/lamda
    result  = gaussianFunction(f1, sig, overlapIndex)
    filters.append(result)
    return filters

def _calculate_amfm(f):
    N1, N2 = f.shape
    # IA = instantaneous amplitude (a_n)
    # IP = instanteneous phase (φ_n)
    # IF = instantanteous frequency (grad φ_n = [grad φ1_n, grad φ2_n])
    IA = np.abs(f)
    IP = np.angle(f)
    IANorm = np.divide(f, IA)
    IFx = np.zeros((N1,N2), np.double)
    IFy = np.zeros((N1,N2), np.double)
    for i in range(1,N1-1):
        for j in range(1,N2-1):
            IFx[i,j] = np.abs(np.arccos(np.real(
                (IANorm[i+1,j]+IANorm[i-1,j]) / (2*IANorm[i,j]) )))
            IFy[i,j] = np.abs(np.arccos(np.real(
                (IANorm[i,j+1]+IANorm[i,j-1]) / (2*IANorm[i,j]) )))
    return IA, IP, IFx, IFy

def _dca(band):
    IA = np.zeros(band[0][0].shape)
    IP = np.zeros(band[0][1].shape)
    IFx = np.zeros(band[0][2].shape)
    IFy = np.zeros(band[0][3].shape)
    w, h = band[0][0].shape
    
    for i in range(w):
        for j in range(h):
            pos = 0
            temp = band[pos][0][i,j] # band[pos].IA[i,j]
            for l in range(len(band)):
                if temp < band[l][0][i,j]: # band[l].IA[i,j]
                    pos = l
                    temp = band[l][0][i,j]
            IA[i,j] = temp
            IP[i,j] = band[pos][1][i,j]
            IFx[i,j] = band[pos][2][i,j]
            IFy[i,j] = band[pos][3][i,j]
    
    return IA, IP, IFx, IFy
    
def amfm_features(f):
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    AMFM = []
    filters = _filterbank()
    hImg = signal.hilbert(f)
    for i, filtre in enumerate(filters):
        filterImg = signal.convolve2d(hImg, np.rot90(filtre), mode='same', 
                                      boundary='fill', fillvalue=0)
        IA, IP, IFx, IFy = _calculate_amfm(filterImg)
        IA = np.nan_to_num(IA)
        IP = np.nan_to_num(IP)
        IFx = np.nan_to_num(IFx)
        IFy = np.nan_to_num(IFy)
        AMFM.append([IA, IP, IFx, IFy])
     
    # Access like this: band[i][0] for IA, band[i][1] for IP,
    # band[i][2] for IFx and band[i][3] for IFy
    high = []
    med = []
    low = []
    dc = []
    for i in range(len(filters)):
        if (i <= 7):
            high.append(AMFM[i])
        elif (i<=23):
            med.append(AMFM[i])
        elif (i<=39):
            low.append(AMFM[i])
        else:
            dc.append(AMFM[i])
    
    IAl, IPl, IFxl, IFyl = _dca(low)
    IAl = (IAl > np.percentile(IAl,50)).astype(np.float64) * IAl
    reconstructionImgDCAl = np.real(IAl * np.cos(IPl))
    H1 = np.histogram(reconstructionImgDCAl, bins=32, density=True)[0]
    
    IAm, IPm, IFxm, IFym = _dca(med)
    IAm = (IAm > np.percentile(IAl,50)).astype(np.float64) * IAm
    reconstructionImgDCAm = np.real(IAm * np.cos(IPm))
    H2 = np.histogram(reconstructionImgDCAm, bins=32, density=True)[0]
    
    IAh, IPh, IFxh, IFyh = _dca(high)
    IAh = (IAh > np.percentile(IAl,50)).astype(np.float64) * IAh
    reconstructionImgDCAh = np.real(IAh * np.cos(IPh))
    H3 = np.histogram(reconstructionImgDCAh, bins=32, density=True)[0]
    
    IAdc, IPdc, IFxdc, IFydc = _dca(dc)
    reconstructionImgDCAdc = np.real(IAdc * np.cos(IPdc))
    H4 = np.histogram(reconstructionImgDCAdc, bins=32, density=True)[0]
    
    features = np.concatenate([H1, H2, H3, H4])
    labels = []
    labels.append(['AMFM_low'+str(i) for i in range(32)])
    labels.append(['AMFM_med'+str(i) for i in range(32)])
    labels.append(['AMFM_high'+str(i) for i in range(32)])
    labels.append(['AMFM_dc'+str(i) for i in range(32)])
    labels = [item for sublist in labels for item in sublist]
    
    warnings.simplefilter(action='default', category=RuntimeWarning)
    
    return features, labels
    
def plotAMFM(f):
    pass
    