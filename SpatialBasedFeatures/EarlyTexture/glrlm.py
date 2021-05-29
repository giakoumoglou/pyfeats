# -*- coding: utf-8 -*-
"""
==============================================================================
@author: https://github.com/szhHr/Gray-Level-Run-Length-Matrix-to-get-image-feature/blob/master/GrayRumatrix.py
@date: Sat May  8 17:00:35 2021
@reference: [30] Gallowway, Texture Analysis using Gray Level Run Lengths
==============================================================================
A8. Gray Level Run Length Matrix
==============================================================================
Inputs:
    - f:        image of dimensions N1 x N2
Outputs:
    - features: 1)Short Run Emphasis, 2)Long Run Emphasis, 3)Gray Level 
                Non-Uniformity/Gray Level Distribution, 4)Run Length 
                Non-Uniformity/Run Length Distribution, 5)Run Percentage,
                6)Low Gray Level Run Emphasis, 7)High Gray Level Run Emphasis,
                8)Short Low Gray Level Emphasis, 9)Short Run High Gray Level 
                Emphasis, 10)Long Run Low Gray Level Emphasis, 11)Long Run 
                High Gray Level Emphasis
==============================================================================
"""
    
import numpy as np
from itertools import groupby

def _apply_over_degree(function, x1, x2):
    if function == np.divide:
        x2 = x2 + 1e-16
    rows, cols, nums = x1.shape
    result = np.ndarray((rows, cols, nums))
    for i in range(nums):
        result[:, :, i] = function(x1[:, :, i], x2)
        result[result == np.inf] = 0
        result[np.isnan(result)] = 0
    return result 
    
def _calculate_ij (rlmatrix):
    gray_level, run_length, _ = rlmatrix.shape
    I, J = np.ogrid[0:gray_level, 0:run_length]
    return I, J+1
    
def _calculate_s(rlmatrix):
    return np.apply_over_axes(np.sum, rlmatrix, axes=(0, 1))[0, 0]

def glrlm(f, theta):
        
    P = np.array(f, np.double)
    x, y = P.shape
    min_pixels = np.min(P)   # the min pixel
    run_length = max(x, y)   # Maximum parade length in pixels
    num_level = np.max(P) - np.min(P) + 1   # Image gray level
        
    deg0 = [val.tolist() for sublist in np.vsplit(P, x) for val in sublist]   # 0deg
    deg90 = [val.tolist() for sublist in np.split(np.transpose(P), y) for val in sublist]   # 90deg
    diags = [P[::-1, :].diagonal(i) for i in range(-P.shape[0]+1, P.shape[1])]   #45deg
    deg45 = [n.tolist() for n in diags]
    Pt = np.rot90(P, 3)   # 135deg
    diags = [Pt[::-1, :].diagonal(i) for i in range(-Pt.shape[0]+1, Pt.shape[1])]
    deg135 = [n.tolist() for n in diags]
        
    def length(l):
        if hasattr(l, '__len__'):
            return np.size(l)
        else:
            i = 0
            for _ in l:
                i += 1
            return i
        
    glrlm = np.zeros((num_level.astype('i'), run_length, len(theta)))   
    for angle in theta:
        for splitvec in range(0, len(eval(angle))):
            flattened = eval(angle)[splitvec]
            answer = []
            for key, iter in groupby(flattened):  
                answer.append((key, length(iter)))   
            for ansIndex in range(0, len(answer)):
                glrlm[int(answer[ansIndex][0]-min_pixels), int(answer[ansIndex][1]-1), theta.index(angle)] += 1 
            
    return glrlm

def glrlm_features(f):
    
    th=['deg0','deg45','deg90','deg135']
    labels = ["GLRLM_ShortRunEmphasis",
              "GLRLM_LongRunEmphasis",
              "GLRLM_GrayLevelNo-Uniformity",
              "GLRLM_RunLengthNonUniformity",
              "GLRLM_RunPercentage",
              "GLRLM_LowGrayLevelRunEmphasis",
              "GLRLM_HighGrayLevelRunEmphasis",
              "GLRLM_Short owGrayLevelEmphasis",
              "GLRLM_ShortRunHighGrayLevelEmphasis",
              "GLRLM_LongRunLowGrayLevelEmphasis",
              "GLRLM_LongRunHighGrayLevelEmphasis"]
    
    rlmatrix = glrlm(f,th)
        
    I, J = _calculate_ij(rlmatrix)
    S = _calculate_s(rlmatrix)
    G = np.apply_over_axes(np.sum, rlmatrix, axes=1)
    R = np.apply_over_axes(np.sum, rlmatrix, axes=0)
        
    features = np.zeros(11,np.double)
    features[0] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.divide, rlmatrix, (J*J)), axes=(0, 1))[0, 0])/S).mean()
    features[1] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.multiply, rlmatrix, (J*J)), axes=(0, 1))[0, 0])/S).mean()
    features[2] = ((np.apply_over_axes(np.sum, (G*G), axes=(0, 1))[0, 0])/S).mean()
    features[3] = ((np.apply_over_axes(np.sum, (R*R), axes=(0, 1))[0, 0])/S).mean()
        
    gray_level, run_length,_ = rlmatrix.shape
    num_voxels = gray_level * run_length
    features[4] = (S/num_voxels).mean()
        
    features[5]= ((np.apply_over_axes(np.sum, _apply_over_degree(np.divide, rlmatrix, (I*I)), axes=(0, 1))[0, 0])/S).mean()
    features[6] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.multiply, rlmatrix, (I*I)), axes=(0, 1))[0, 0])/S).mean()
    features[7] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.divide, rlmatrix, (I*I*J*J)), axes=(0, 1))[0, 0])/S).mean()
        
    temp = _apply_over_degree(np.multiply, rlmatrix, (I*I))
    features[8] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.divide, temp, (J*J)), axes=(0, 1))[0, 0])/S).mean()
        
    temp = _apply_over_degree(np.multiply, rlmatrix, (J*J))
    features[9] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.divide, temp, (J*J)), axes=(0, 1))[0, 0])/S).mean()
    features[10] = ((np.apply_over_axes(np.sum, _apply_over_degree(np.multiply, rlmatrix, (I*I*J*J)), axes=(0, 1))[0, 0])/S).mean()
        
    return features, labels    