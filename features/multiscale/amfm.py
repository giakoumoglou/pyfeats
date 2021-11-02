# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Kyriacos Constantinou
@author: Ioannis Constantinou,
@author: Marios S. Pattichis
@author: Constantinos S. Pattichis
@reference: Murray, An AM-FM model for Motion Estimation in Atherosclerotic Plaque Videos
            Murray, Multiscale AMFM Demodulation and Image Reconstruction methods with Improved Accuracy 
            Pattichis, Medical Image Analysis Using AM-FM Models and Methods
==============================================================================
"""

import numpy as np
from  scipy import signal
import warnings

def _gabor_kernel_2D(theta, lamda, gamma, bandwidth, phase, overlapIndex):
    qFactor = (1/np.pi) * np.sqrt( (np.log(overlapIndex)/2) ) *  \
                    ( (2**bandwidth + 1) / (2**bandwidth - 1) )
    sigma = lamda*qFactor
    n = np.ceil(4*sigma)
    [x,y] = np.mgrid[-n:(n+2),-n:(n+2)]
    xTheta = x * np.cos(theta) + y * np.sin(theta)
    yTheta = -x * np.sin(theta) + y * np.cos(theta)
    gaussian = np.exp(-(( xTheta**2) + gamma**2.* (yTheta**2))/(2*sigma**2))
    res = gaussian * np.cos(2*np.pi*xTheta/lamda +  phase)
    maxFft = abs(np.fft.fft2(res)).max()
    normalize = np.fft.fft2(res)/maxFft
    result = np.real(np.fft.ifft2(normalize))
    return result, sigma
    
def _gaussian_function(f0, s0, overlapIndex):
    over = np.sqrt(2*np.log(1/overlapIndex))
    sigma = s0*over/(s0*f0 - over)
    n = np.ceil(2*sigma)
    [x,y] = np.mgrid[-n:(n+2),-n:(n+2)]
    res = np.exp(-1/2*(x**2 + y**2)/sigma**2)
    res = res / res.sum()
    return res
 
def _filterbank():
    '''
    Returns
    -------
    filter : list
        List of 41 filters for AM-FM multi-scale analysis.
    '''
    
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
            result, sig = _gabor_kernel_2D(theta[th], lamda, gamma, bandwidth, phase, 1/overlapIndex);
            filters .append(result)
        lamda = lamda0 * 2**bandwidth
    # Add DC Filter
    f1      = 2*np.pi/lamda
    result  = _gaussian_function(f1, sig, overlapIndex)
    filters.append(result)
    return filters

def _calculate_amfm(f):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.

    Returns
    -------
    IA : numpy ndarray
        instantaneous amplitude (a_n).
    IP : numpy ndarray
        instanteneous phase (φ_n).
    IFx : numpy ndarray
        instantanteous frequency (grad φ1_n).
    IFy : numpy ndarray
        instantanteous frequency (grad φ2_n).
    '''
    N1, N2 = f.shape
    IA = np.abs(f)
    IP = np.angle(f)
    IANorm = np.divide(f, IA+1e-16)
    IFx = np.zeros((N1,N2), np.double)
    IFy = np.zeros((N1,N2), np.double)
    for i in range(1,N1-1):
        for j in range(1,N2-1):
            IFx[i,j] = np.abs(np.arccos(np.real((IANorm[i+1,j]+IANorm[i-1,j]) / (2*IANorm[i,j]))))
            IFy[i,j] = np.abs(np.arccos(np.real((IANorm[i,j+1]+IANorm[i,j-1]) / (2*IANorm[i,j]))))
    return IA, IP, IFx, IFy

def _dca(band):
    '''
    Parameters
    ----------
    band : list
        The band. A list of IA, IP, IFx, IFy.

    Returns
    -------
    IA : list
        Max instantaneous amplitude for given band.
    IP : list
        Max instanteneous phase for given band.
    IFx : list
        Max instantanteous frequency for given band.
    IFy : list
        Max instantanteous frequency for given band.
    '''
    
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
    
def amfm_features(f, bins=32):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    bins: int, optional
        Bins for the calculated histogram. The default is 32.

    Returns
    -------
    features : numpy ndarray
        Histogram of IA, IP, IFx, IFy as a concatenated vector.
    labels : list
        Labels of features.
    '''
    

    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    AMFM = []
    filters = _filterbank()
    f_hilbert = signal.hilbert(f)
    
    #mask_c = _image_xor(mask)
    #mask_conv = []
    #for filtre in filters:
    #    oneskernel = np.ones(filtre.shape)
    #    temp = signal.convolve2d(mask_c, oneskernel,'same')
    #    temp = np.abs(np.sign(temp)-1)
    #    mask_conv.append(temp)
        
    for i, filtre in enumerate(filters):
        f_filtered = signal.convolve2d(f_hilbert, np.rot90(filtre), mode='same', boundary='fill', fillvalue=0)
        #f_filtered = f_filtered * mask_conv[i]
        IA, IP, IFx, IFy = _calculate_amfm(f_filtered)
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
    H1 = np.histogram(reconstructionImgDCAl, bins=bins, density=True)[0]
    
    IAm, IPm, IFxm, IFym = _dca(med)
    IAm = (IAm > np.percentile(IAl,50)).astype(np.float64) * IAm
    reconstructionImgDCAm = np.real(IAm * np.cos(IPm))
    H2 = np.histogram(reconstructionImgDCAm, bins=bins, density=True)[0]
    
    IAh, IPh, IFxh, IFyh = _dca(high)
    IAh = (IAh > np.percentile(IAl,50)).astype(np.float64) * IAh
    reconstructionImgDCAh = np.real(IAh * np.cos(IPh))
    H3 = np.histogram(reconstructionImgDCAh, bins=bins, density=True)[0]
    
    IAdc, IPdc, IFxdc, IFydc = _dca(dc)
    reconstructionImgDCAdc = np.real(IAdc * np.cos(IPdc))
    H4 = np.histogram(reconstructionImgDCAdc, bins=bins, density=True)[0]
    
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
    