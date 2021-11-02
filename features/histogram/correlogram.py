# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Nikolaos Giakoumoglou
@date: Fri May 14 10:33:48 2021
==============================================================================
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

def correlogram(f, mask, bins_digitize = 32, bins_hist = 32, flatten=False):
    ''' 
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    bins_digitize : int, optional
         Number of bins for discrete distances and thetas. The default is 32.
    bins_hist : int, optional
        Number of bins for histogram. The default is 32.
    flatten : bool, optional
        Return correlogram as 1d array if True or 2d array if False. The 
        default is False.

    Returns
    -------
    Hd : numpy ndarray
        Correlogram for distance.
    Ht : numpy ndarray
        Correlogram for angles.
    labels : list
        Labels of features.
    '''
    
    if mask is None:
        mask = np.ones(f.shape) 
    
    Ng = 256
    
    # Step 1: Find center pixel
    N1, N2 = f.shape
    c = [np.floor(N1/2), np.floor(N2/2)]
    
    # Step 2.1: Find distance of each pixel from center pixels
    D = np.zeros((N1,N2), np.double) * np.nan
    for i in range(N1):
        for j in range(N2):
            D[i,j] = distance.euclidean(c, [i,j])
    D[mask == 0] = 0
    D /= D.max() # normalise to [0,1]
    D = np.digitize(D, bins=np.arange(0,1,1/bins_digitize), 
                    right=False).astype(np.float)
    D[mask == 0] = np.nan
    
    # Step 2.2 Make histogram
    Hd = np.zeros((bins_digitize,bins_hist), np.double)
    for b in range(bins_digitize):
        Hd[b,:] = np.histogram(f[D==(b+1)], bins=bins_hist, range=(0,Ng-1))[0]
        
        
    # Step 3.1: Find angle of each pixel from center pixel
    T = np.zeros((N1,N2), np.double) * np.nan
    for i in range(N1):
        for j in range(N2):
            x = j - c[1]
            y = -(i - c[0])
            angle = np.arctan2(y,x)
            T[i,j] = np.degrees(angle)
            
    # angles between -180 and 180 degrees
    T = np.digitize(T, bins=np.arange(-180,180,360/bins_digitize), 
                    right=False).astype(np.float)
    T[mask == 0] = np.nan
    
    # Step 3.2: Make histogram
    Ht = np.zeros((bins_digitize,bins_hist), np.double)
    for b in range(bins_digitize):
        Ht[b,:] = np.histogram(f[T==(b+1)], bins=bins_hist, range=(0,Ng-1))[0]

    # Step 4: Create label
    labels = []
    for i in range(bins_digitize):
        for j in range(bins_hist):
            labels.append('Correlogram_'+str(i)+'_'+str(j))
    
    # Step 5: Return flatten or 2d array
    if (flatten == True):
        return Hd.flatten(), Ht.flatten(), labels
    else:
        return Hd, Ht, labels

def plot_correlogram(f, mask, bins_digitize = 32, bins_hist = 32, name=''):
    Hd, Ht, _ = correlogram(f, mask, bins_digitize, bins_hist, False)
    if name != '':
        name = '('+name+')'
    fig, ax = plt.subplots()
    ax.imshow(Hd, cmap='hot')
    ax.set_title('Correlogram '+str(Hd.shape[0])+'x'+str(Hd.shape[1])+': Distances'
                 +' '+name)
    
    fig, ax = plt.subplots()
    ax.imshow(Ht, cmap='hot')
    ax.set_title('Correlogram '+str(Ht.shape[0])+'x'+str(Ht.shape[1])+': Angles'
                 +' '+name)
    plt.show()

