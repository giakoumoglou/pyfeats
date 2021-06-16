# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Thu May 20 18:24:36 2021
@reference: [53] Hamilton, Fast automated cell phenotype image classification
==============================================================================
B.5 Threshold Adjacency Statistis (TAS)
==============================================================================
Inputs:
    - f:         image of dimensions N1 x N2
Outputs:
    - features:  feature values
==============================================================================
"""
import mahotas

def tas_features(f):
    features = mahotas.features.pftas(f)
    labels = ['TAS' + str(i) for i in range(features.shape[0])]
    return features, labels