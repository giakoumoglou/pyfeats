# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Thu May 20 18:37:25 2021
@reference: Hu, Visual Pattern Recognition by Moment Invariants
==============================================================================
Hu's moments
==============================================================================
Inputs:
    - f:         image of dimensions N1 x N2
Outputs:
    - features:  7 Hu' invariants
==============================================================================
"""

import cv2

def hu_moments(f):
    features = cv2.HuMoments(cv2.moments(f)).flatten()
    labels = ['Hu_Moment_' + str(i) for i in range(features.shape[0])]
    return features, labels