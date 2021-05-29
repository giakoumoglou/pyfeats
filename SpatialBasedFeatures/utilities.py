# -*- coding: utf-8 -*-

import numpy as np
import math

__all__ = ['_energy', '_entropy', '_next_power_of_two', '_pad_image', 
           '_mean_std', 'gaborKernel2D', 'gaussianFunction','_fft2']

def _energy(x):
    return np.multiply(x,x).sum()

def _entropy(x):
    return -np.multiply(x, np.log(x+1e-16)).sum()

def _next_power_of_two(n):
    math.ceil(math.log(n,2))
    return 2 ** math.ceil(math.log(n,2))

def _pad_image(f):
    N1, N2 = f.shape
    N1_deficit = _next_power_of_two(N1) - N1
    N2_deficit = _next_power_of_two(N2) - N2
    f2 = np.pad(f, ((math.floor(N1_deficit/2),N1_deficit-math.floor(N1_deficit/2)), 
                    (math.floor(N2_deficit/2),N2_deficit-math.floor(N2_deficit/2))), 
                mode='constant')
    return f2
    
def _mean_std(D):
    N1, N2 = D.shape
    mi = sum(sum(abs(D))) / (N1*N2)
    sigma = sum(sum(abs(D-mi) ** 2)) / (N1*N2)
    return mi, sigma

def gaborKernel2D(theta, lamda, gamma, bandwidth, phase, 
                  overlapIndex):
    qFactor = (1/np.pi) * np.sqrt( (np.log(overlapIndex)/2) ) *  \
                    ( (2**bandwidth + 1) / (2**bandwidth - 1) )
    sigma = lamda*qFactor
    n = np.ceil(4*sigma)
    [x,y] = np.mgrid[-n:(n+2),-n:(n+2)]
    xTheta = x * np.cos(theta) + y * np.sin(theta)
    yTheta = -x * np.sin(theta) + y * np.cos(theta)
    gaussian = np.exp(-(( xTheta**2) + gamma**2.* (yTheta**2))/(2*sigma**2))
    res = gaussian * np.cos(2*np.pi*xTheta/lamda +  phase)
    maxFft = abs(np.fft.fft2(res)).max()
    normalize = np.fft.fft2(res)/maxFft
    result = np.real(np.fft.ifft2(normalize))
    return result, sigma
    
def gaussianFunction(f0, s0, overlapIndex):
    over = np.sqrt(2*np.log(1/overlapIndex))
    sigma = s0*over/(s0*f0 - over)
    n = np.ceil(2*sigma)
    [x,y] = np.mgrid[-n:(n+2),-n:(n+2)]
    res = np.exp(-1/2*(x**2 + y**2)/sigma**2)
    res = res / res.sum()
    return res

def _image_xor(f):
    # Turn "0" to "1" and vice versa: XOR with image consisting of "1"s
    f = f.astype(np.uint8)
    mask = np.ones(f.shape, np.uint8)
    out = np.zeros(f.shape, np.uint8)
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            out[i,j] = f[i,j] ^ mask[i,j]
    return out

def _fft2(x,msk):
    r,c = x.shape
    f1 = np.zeros((r,c), np.complex64)
    
    for i in range(1,c+1):
      sp = 0
      ep = 0
      j = 1
      while (j < r):
          while (msk[j-1,i-1]==1) & (j < r):
              if (sp == 0):
                  sp = j
              j += 1
          if (sp > 0) & (ep == 0):
              if (j < r):
                  ep = j-1
              else:
                  ep = j      
              f1[(sp-1):(ep),i-1] = np.fft.fft(x[(sp-1):(ep),i-1])
              sp = 0
              ep = 0
          while (msk[j-1,i-1] == 0) & (j < r):
              j += 1

    f1 = f1.T
    msk = msk.T
    
    for i in range(1,r+1):
        sp = 0
        ep = 0
        j = 1
        while (j < c):
          while (msk[j-1,i-1] == 1) & (j < c):
              if (sp == 0):
                  sp = j
              j += 1
          if (sp > 0) & (ep == 0):
              if (j < r):
                  ep = j-1
              else:
                  ep = j     
              f1[(sp-1):(ep),i-1] = np.fft.fft(f1[(sp-1):(ep),i-1])
              sp = 0
              ep = 0
          while (msk[j-1,i-1] == 0) & (j < c):
              j += 1

    f1 = f1.T
    msk = msk.T
    return f1