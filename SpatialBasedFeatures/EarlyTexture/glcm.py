# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Thu May  6 19:10:11 2021
@reference: [7] Haralick, Textural Features for Image Classification
==============================================================================
B.1 Gray Level Co-occurence Matrix/Spatial Gray level Difference Matrix
==============================================================================
Inputs:
    - f:             image of dimensions N1 x N2
    - d:             distance to calculate Co-occurence matrix (default d=1)
    - th:            angle to calculate Co-occurence matrix 
                     (default th=[0,45,90,135])
    - ignore_zeros:  ignoore zeros due to mask (default True)
Outputs:
    - features:      Haralick's 1)Angular Second Moment, 2)Contrast, 
                     3)Correlation, 4)Sum of Squares: Variance, 5)Inverse 
                     Difference Moment 6)Sum Average, 7)Sum Variance, 8)Sum 
                     Entropy, 9)Entropy, 10)Difference Variance, 11)Difference 
                     Entropy, 12)Information Measure of Correlation 1, 
                     13)Information Measure of Correlation 2, 14)Maximal 
                     Correlation Coefficient    
==============================================================================
"""
import numpy as np
import mahotas

def glcm_features(f, ignore_zeros=True):
    
    # 1) Labels
    labels = ["GLCM_ASM", "GLCM_Contrast", "GLCM_Correlation",
              "GLCM_SumOfSquaresVariance", "GLCM_InverseDifferenceMoment",
               "GLCM_SumAverage", "GLCM_SumVariance", "GLCM_SumEntropy",
               "GLCM_Entropy", "GLCM_DifferenceVariance",
               "GLCM_DifferenceEntropy", "GLCM_Information1",
               "GLCM_Information2", "GLCM_MaximalCorrelationCoefficient"]
    labels_mean = [label + "_Mean" for label in labels]
    labels_range = [label + "_Range" for label in labels]
    
    # 2) Parameters
    f = f.astype(np.uint8)
    
    # 3) Calculate Features: Mean and Range
    features = mahotas.features.haralick(f, 
                                         ignore_zeros=ignore_zeros, 
                                         compute_14th_feature=True,
                                         return_mean_ptp=True)
    features_mean = features[0:14]
    features_range = features[14:]
    
    return features_mean, features_range, labels_mean, labels_range
