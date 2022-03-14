# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Thu May 20 11:51:54 2021
==============================================================================
"""

from skimage import feature
import matplotlib.pyplot as plt

def hog_features(f, ppc=8, cpb=3):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    ppc : int, optional
        Pixels per cell. The default is 8.
    cpb : int, optional
        Cells per block. The default is 3.

    Returns
    -------
    fd : numpy ndarray
        Histogram of Oriented Gradients flattened.
    labels : list
        Labels of features.
    '''
    
    fd, _ = feature.hog(f, orientations=9, pixels_per_cell=(ppc,ppc), 
                    cells_per_block=(cpb,cpb), block_norm='L2', visualize=True)
    labels = [('HOS_ppc_' + str(ppc) + '_cpb' + str(cpb) + '_' + str(i)) for i in range(fd.shape[0])]
    return fd, labels

def plot_hog(f, ppc=16, cpb=3, name=''):
    _, hog_image = feature.hog(f, orientations=9, pixels_per_cell=(ppc,ppc), 
                    cells_per_block=(cpb,cpb), block_norm='L2', visualize=True)
    if name != '':
        name = '('+name+')'
    plt.imshow(hog_image)     
    plt.title('Histogram of Oriented Gradients: Image ' + str(name))  
    plt.show()
    