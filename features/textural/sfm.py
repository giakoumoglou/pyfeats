# -*- coding: utf-8 -*-
"""
==============================================================================
@author: Christos Loizou
@date: Fri May  7 16:20:47 2021
@reference: Wu, Statistical Feature Matrix for Texture Analysis
==============================================================================
"""
import numpy as np

def con_cov_dss(f, mask, Lr, Lc, Ng=256):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else
    Lr : int
        Parameters of SFM.
    Lc : int
        Parameters of SFM.
    Ng : int, optional
        Image number of gray values. The default is 256.

    Returns
    -------
    CON : numpy ndarray
    COV : numpy ndarray
    DSS : numpy ndarray
    '''
    
    N1, N2 = f.shape
    CON = np.zeros((Lr+1,2*Lc+1),np.double) # delta contrast
    COV = np.zeros((Lr+1,2*Lc+1),np.double) # deltra covariance
    DSS = np.zeros((Lr+1,2*Lc+1),np.double) # delta dissimilarity
        
    for drow in range(Lr+1):
        for dcol in range(-Lc,Lc+1):
                
            # Define coords in statistical feature matrix
            row = drow
            col = dcol + Lc
                
            # Shift image by drow, dcol := f_d
            f_d = np.ones((N1,N2),np.double) * (Ng - 1)
            mask_d = np.zeros((N1,N2),np.double)
            if dcol < 0:
                f_d[0:(N1-drow),(-dcol):(N2)] = f[(drow):(N1),0:(N2+dcol)]
                mask_d[0:(N1-drow),(-dcol):(N2)] = mask[(drow):(N1),0:(N2+dcol)]
            else:
                f_d[0:(N1-drow),0:(N2-dcol)] = f[(drow):(N1),(dcol):(N2)]
                mask_d[0:(N1-drow),0:(N2-dcol)] = mask[(drow):(N1),(dcol):(N2)]
                            
            # Find mask to exclude pixels outside mask 
            mask_common = np.multiply(mask,mask_d)
                
            n = sum(sum(mask_common))
            f_mean = sum(sum(np.multiply(f,mask_common)))/n;
                
            # Compute CON, COV, DSS
            CON[row,col] = sum(sum(np.multiply((f-f_d),mask_common)**2))/n
            temp = np.multiply((f-f_mean),(f_d-f_mean))
            COV[row,col] = sum(sum(np.multiply(mask_common,temp)))/n
            DSS[row,col] = sum(sum(abs(np.multiply((f-f_d),mask_common))))/n
        
    # 4) Set first half on first row to zero to preserve symmetry
    CON[0,0:Lc] = 0
    COV[0,0:Lc] = 0
    DSS[0,0:Lc] = 0
    
    return CON, COV, DSS

def sfm_features(f, mask, Lr=4, Lc=4):
    '''  
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    Lr : int, optional
        Parameters of SFM. The default is 4.
    Lc : int, optional
        Parameters of SFM. The default is 4.

    Returns
    -------
    features : numpy ndarray
        1)Coarseness, 2) Contrast, 3)Periodicity, 4)Roughness.
    labels : list
        Labels of features.
    '''
    
    if mask is None:
        mask = np.ones(f.shape)
        
    # 1) Labels
    labels = ["SFM_Coarseness","SFM_Contrast","SFM_Periodicity","SFM_Roughness"]
    
    # 2) Parameters
    f = np.array(f, np.double)
    mask = np.array(mask, np.double)
    Ng = 256
         
    # 3) Calculate CON, COV, DSS
    CON, COV, DSS = con_cov_dss(f, mask, Lr, Lc, Ng)
        
    # 4) CaLculate features
    features = np.zeros(4,np.double)
    r = min([Lr,Lc]);
    features[0] = 100*(r+1)*(2*r)/sum(sum(DSS[0:(r+1),(Lc-r):(1+Lc+r)]))
    features[1] = np.sqrt(sum(sum(CON[0:2,(Lc-1):(Lc+2)]))/2)
    DSScol = DSS.reshape(-1)
    DSScol[0:(Lc+1)] = 0
    DSSmean = sum(sum(DSS)) / ((Lr+1)*(2*Lc+1)-(Lc+1))
    features[2] = (DSSmean-min(DSScol[np.nonzero(DSScol)]))/DSSmean  
    lgDSSr = np.log(DSS[0,(Lc+1):(2*Lc+1)] + 1e-16)
    lgDSSc = np.log(DSS[1:(Lr+1),(Lc)] + 1e-16)
    lgLrd = np.log(np.arange(1,Lc+1))
    lgLcd = np.log(np.arange(1,Lr+1))
    Hr = (Lc*sum(np.multiply(lgDSSr,lgLrd)) - sum(lgLrd)*sum(lgDSSr)) / (Lc*sum(lgLrd**2)-sum(lgLrd)**2)
    Hc = (Lr*sum(np.multiply(lgDSSc,lgLcd)) - sum(lgLcd)*sum(lgDSSc)) / (Lr*sum(lgLcd**2)-sum(lgLcd)**2)
    features[3] =(6-Hr-Hc)/2;
        
    return features, labels