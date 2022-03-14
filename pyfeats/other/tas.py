# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Thu May 20 18:24:36 2021
@reference: Hamilton, Fast automated cell phenotype image classification
==============================================================================
Threshold Adjacency Statistis (TAS)
==============================================================================
Inputs:
    - f:         image of dimensions N1 x N2
Outputs:
    - features:  feature values
==============================================================================
"""
import mahotas

def tas_features(f):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.

    Returns
    -------
    features : numpy ndarray
        Feature values.
    labels : list
        Labels of features.
    '''    

    features = mahotas.features.pftas(f)
    labels = ['TAS' + str(i) for i in range(features.shape[0])]
    return features, labels