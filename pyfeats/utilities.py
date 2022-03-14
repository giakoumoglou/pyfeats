# -*- coding: utf-8 -*-

import numpy as np
import math
import cv2

__all__ = ['_energy', '_entropy', '_next_power_of_two', '_pad_image_power_2',
           'pad_image', '_zero_runs']

def _energy(x):
    return np.multiply(x,x).sum()

def _entropy(x):
    return -np.multiply(x, np.log(x+1e-16)).sum()

def pad_image(f, pad=2, fill_val=255):
    TDLU=[1, 1, 1, 1]  #top, down, left, right pad
    out = f.copy()
    for _ in range(pad):
        out = cv2.copyMakeBorder(out, TDLU[0], TDLU[1], TDLU[2], TDLU[3],\
                                 cv2.BORDER_CONSTANT, None, fill_val)
    return out

def _next_power_of_two(n):
    math.ceil(math.log(n,2))
    return 2 ** math.ceil(math.log(n,2))

def _pad_image_power_2(f):
    N1, N2 = f.shape
    N1_deficit = _next_power_of_two(N1) - N1
    N2_deficit = _next_power_of_two(N2) - N2
    f2 = np.pad(f, ((math.floor(N1_deficit/2),N1_deficit-math.floor(N1_deficit/2)), 
                    (math.floor(N2_deficit/2),N2_deficit-math.floor(N2_deficit/2))), 
                mode='constant')
    return f2

def _image_xor(f):
    # Turn "0" to "1" and vice versa: XOR with image consisting of "1"s
    f = f.astype(np.uint8)
    mask = np.ones(f.shape, np.uint8)
    out = np.zeros(f.shape, np.uint8)
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            out[i,j] = f[i,j] ^ mask[i,j]
    return out

def _zero_runs(img, dimension):
    out = []
    if dimension == 0:
        dim0 = img.shape[0] 
        for dim in range(dim0):
            a = img[dim, :]
            iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
            absdiff = np.abs(np.diff(iszero))
            ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
            if ranges.size == 0:
                continue
            else:
                out.append(ranges[0][1]-ranges[0][0]+1)
    elif dimension == 1:
        dim1 = img.shape[1] 
        for dim in range(dim1):
            a = img[:, dim]
            iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
            absdiff = np.abs(np.diff(iszero))
            ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
            if ranges.size == 0:
                continue
            else:
                out.append(ranges[0][1]-ranges[0][0]+1)
    else:
        print("Error! Dimension must be 0 or 1 in an image.")
    return max(out)

    #mask_c = _image_xor(mask)
    #features[0] = _zero_runs(mask_c, 1)
    #features[1] = _zero_runs(mask_c, 0)  