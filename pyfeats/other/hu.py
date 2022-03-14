# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Thu May 20 18:37:25 2021
@reference: Hu, Visual Pattern Recognition by Moment Invariants
==============================================================================
"""

import cv2

def hu_moments(f):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.

    Returns
    -------
    features : numpy ndarray
        Hu's moments.
    labels : list
        Labels of features.
    '''
    
    features = cv2.HuMoments(cv2.moments(f)).flatten()
    labels = ['Hu_Moment_' + str(i) for i in range(features.shape[0])]
    return features, labels