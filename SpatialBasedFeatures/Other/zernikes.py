# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Thu May 20 13:46:57 2021
@reference: [54] Teague, Image analysis via the general theory of moments
==============================================================================
C.12 Zernike's Moments
==============================================================================
Inputs:
    - f:             image of dimensions N1 x N2
    - radius:        radius to calculate Zernikes moments
Outputs:
    - features:      Zernikes moments
==============================================================================
"""

import mahotas

def zernikes_moments(f, radius=9):
    features = mahotas.features.zernike_moments(f, radius)
    labels = ['Zernikes_Moments_radius_' + str(radius) + '_' + str(i) for i in range(features.shape[0])]
    return features, labels