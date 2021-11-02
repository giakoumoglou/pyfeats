# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Thu May 20 13:46:57 2021
@reference: Teague, Image analysis via the general theory of moments
==============================================================================
"""

import mahotas

def zernikes_moments(f, radius=9):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    radius : int, optional
        Radius to calculate Zernikes moments. The default is 9.

    Returns
    -------
    features : numpy ndarray
        Zernikes' moments.
    labels : list
        Labels of features.
    '''
    
    features = mahotas.features.zernike_moments(f, radius)
    labels = ['Zernikes_Moments_radius_' + str(radius) + '_' + str(i) for i in range(features.shape[0])]
    return features, labels