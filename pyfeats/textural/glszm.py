# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Thu May 20 21:16:32 2021
@reference: Thibault, Texture Indexes and Gray Level Size Zone Matrix 
            Application to Cell Nuclei Classification
==============================================================================
Note that an image and a padded image with "0" give different outputs for
many features since Ns is different (even though if we remove zeros e.g.
ps = ps[ps!=0]) and as a result jvector changes outputs.
==============================================================================
"""

import numpy as np
from skimage import measure
import warnings

def glszm(f, mask, connectivity=1):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    connectivity: Maximum number of orthogonal hops to consider a pixel/voxel 
        as a neighbor. Accepted values are ranging from 1 to input.ndim. If 
        None, a full connectivity of input.ndim is used.

    Returns
    -------
    GLSZM : numpy ndarray
        GLSZ Matrix.
    '''
    if connectivity == None:
        pass
    elif connectivity > f.ndim:
        connectivity = f.ndim
        warnings.warn('Accepted values for connectivity are ranging from 1 to f.ndim. Changed to f.ndim')
    elif connectivity < 1:
        connectivity = 1
        warnings.warn('Accepted values for connectivity are ranging from 1 to f.ndim. Changed to 1')
    
    Ng = 256
    levels=np.arange(0,Ng)

    Ns = f.shape[0] * f.shape[1] + 1
    GLSZM = np.zeros((Ng-1,Ns), np.double)
    
    temp = f.copy()
    for i in range(Ng-1):
        temp[f!=levels[i]] = 0
        temp[f==levels[i]] = 1
        connected_components = measure.label(temp, connectivity=connectivity)
        connected_components = connected_components * mask
        nZone = len(np.unique(connected_components))
        for j in range(nZone):
            col = np.count_nonzero(connected_components==j)
            GLSZM[i,col] += 1
            
    return GLSZM
            
def glszm_features(f, mask):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else.

    Returns
    -------
    features : numpy ndarray
        1)Small Zone Emphasis, 2)Large Zone Emphasis,
        3)Gray Level Nonuniformity, 4)Zone Size Nonuniformit,
        5)Zone Percentage, 6)Low Gra yLeveL Zone Emphasis,
        7)High Gray Level Zone Emphasis, 8)Small Zone Low Gray 
        Level Emphasis, 9)Small Zone High Gray LeveL Emphasis, 
        10)Large Zone Lo wGray Level Emphassis, 11)Large Zone High 
        Gray Level Emphasis, 12)Gray Level Variance,
        13)Zone Size Variance, 14)Zone Size Entropy.
    labels : list
        Labels of features.
    '''
    
    if mask is None:
        mask = np.ones(f.shape)
        
    labels = ['GLSZM_SmallZoneEmphasis', 'GLSZM_LargeZoneEmphasis',
              'GLSZM_GrayLevelNonuniformity', 'GLSZM_ZoneSizeNonuniformity',
              'GLSZM_ZonePercentage', 'GLSZM_LowGrayLeveLZoneEmphasis',
              'GLSZM_HighGrayLevelZoneEmphasis', 'GLSZM_SmallZoneLowGrayLevelEmphasis',
              'GLSZM_SmallZoneHighGrayLevelEmphasis', 'GLSZM_LargeZoneLowGrayLevelEmphassis',
              'GLSZM_LargeZoneHighGrayLevelEmphasis', 'GLSZM_GrayLevelVariance',
              'GLSZM_ZoneSizeVariance','GLSZM_ZoneSizeEntropy']
    
    P = glszm(f, mask)
    # FIXME
    #idx = np.argwhere(np.all(P[..., :] == 0, axis=0))
    #P = np.delete(P, idx, axis=1)
    
    p = P / P.sum()

    Ng, Ns = p.shape
    pg = np.sum(p,1) # Gray-Level Sum [Ng x 1]
    ps = np.sum(p,0) # Zone-Size Sum  [Ns x 1]
    jvector = np.arange(1,Ns+1) 
    ivector = np.arange(1,Ng+1)     
    Nz = np.sum(P, (0,1))
    Np = np.sum(ps * jvector, 0)
    [imat,jmat] = np.meshgrid(jvector,ivector)
        
    features = np.zeros(14, np.double)
    features[0] = np.dot(ps,((1/(jvector+1e-16))**2))
    features[1]= np.dot(ps, jvector ** 2) 
    features[2] = (pg**2).sum()
    features[3] = (ps**2).sum()
    features[4] = Nz / Np
    features[5] = np.dot(pg, 1/(ivector+1e-16)**2) 
    features[6] = np.dot(pg, ivector ** 2)
    features[7] =  np.multiply(p, 
                    np.multiply(1/(jmat+1e-16)**2,1/(imat+1e-16)**2)).sum()
    features[8] = np.multiply(p, 
                    np.multiply(jmat**2,1/(imat+1e-16)**2)).sum()
    features[9] = np.multiply(p, 
                    np.multiply(1/(jmat+1e-16)**2,imat**2)).sum()
    features[10] = np.multiply(p, np.multiply(jmat**2,imat**2)).sum()
    meang = np.dot(pg,ivector)/(Ng*Ns) 
    features[11] = ((np.multiply(p, jmat) - meang) ** 2).sum() / (Ng*Ns)
    means = np.dot(ps,jvector)/(Ng*Ns)
    features[12] = ((np.multiply(p, imat) - means) ** 2).sum() / (Ng*Ns)
    features[13] = np.multiply(p, np.log2(p+1e-16)).sum()

    return features, labels