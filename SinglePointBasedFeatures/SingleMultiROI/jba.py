# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Sun May  9 21:46:00 2021
==============================================================================
Jaxtaluminal Black Area (JBA)
Find area near lumen with luminocity < 25. Area is defined as perc% near the
perimeter of lumen
==============================================================================
Inputs:
    - img:             image 
    - mask:            2D array with 1 inside ROI [int32]
    - perimeter_lumen: 2D array with 1 for pixels in perimeter of ROI that
                       is closest to lumen
    - perc:            percentage of pixels in lumen area to consider for JBA
    - area_mm:         pixel density in mm2 (default=1)
Outputs:
    - features:        Jaxtaluminal Black Area (JBA)
==============================================================================
"""
import numpy as np
from skimage import color, img_as_ubyte
from scipy.spatial.distance import cdist

def _get_perc_ROI(mask, perimeter_lumen, perc):
    dist = np.empty(mask.shape)
    dist[:] = np.inf
    II = np.argwhere(mask)
    JJ = np.argwhere(perimeter_lumen)
    K = tuple(II.T)
    dist[K] = cdist(II, JJ).min(axis=1, initial=np.inf)     
    percPixels = np.fix(perc * np.count_nonzero(mask) ).astype('i')
    def get_indices_of_k_smallest(arr, k):
        idx = np.argpartition(arr.ravel(), k)
        return tuple(np.array(np.unravel_index(idx, arr.shape))[:, range(min(k, 0), max(k, 0))])
    idx = get_indices_of_k_smallest(dist, percPixels)
    idx = np.array(idx,dtype=np.int32).T          
    area_lumen = np.zeros(mask.shape, dtype=np.int32)
    for i in range(idx.shape[0]):
        area_lumen[idx[i,0],idx[i,1]] = 1 
    return area_lumen

def jba_feature(img, mask, perimeter_lumen, perc, area_mm=1):
    img = img_as_ubyte(color.rgb2gray(img))
    area_lumen = _get_perc_ROI(mask, perimeter_lumen, perc)  
    img_ravel = img.ravel()
    area_lumen_ravel = area_lumen.ravel()
    jba = img_ravel[area_lumen_ravel.astype(bool)] # keep only the values of image inside  the mask
    return (jba<25).sum(), ['JBA']
