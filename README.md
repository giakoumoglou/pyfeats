<p align="center">
  <img width="375" height="130" src="https://github.com/giakou4/features/blob/main/demo/data/logo.jpg">
</p>

# PyFeats

Open source software for image feature extraction

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Downloads](https://pepy.tech/badge/pyfeats)](https://pepy.tech/project/pyfeats)
[![Downloads](https://pepy.tech/badge/pyfeats/month)](https://pepy.tech/project/pyfeats)
[![Downloads](https://pepy.tech/badge/pyfeats/week)](https://pepy.tech/project/pyfeats)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/giakou4/Features/LICENSE)
[![PyPi](https://badgen.net/badge/icon/pypi?icon=pypi&label)](https://pypi.org/project/pyfeats/)

## 1. Install through pip

Now the project can be found at https://pypi.org/project/pyfeats/.
Install using the following command:
```console
pip install pyfeats
```
Use calling:
```python
import pyfeats
```

## 2. Features

### 2.1 Textural Features
1. First Order Statistics/Statistical Features (FOS/SF)
2. Gray Level Co-occurence Matrix (GLCM/SGLDM)
3. Gray Level Difference Statistics (GLDS)
4. Neighborhood Gray Tone Difference Matrix (NGTDM)
5. Statistical Feature Matrix (SFM)
6. Law's Texture Energy Measures (LTE/TEM)
7. Fractal Dimension Texture Analysis (FDTA)
8. Gray Level Run Length Matrix (GLRLM)
9. Fourier Power Spectrum (FPS)
10. Shape Parameters
11. Gray Level Size Zone Matrix (GLSZM)
12. Higher Order Spectra (HOS)
13. Local Binary Pattern (LPB)

### 2.2 Morphological Features
1. Grayscale Morphological Analysis
2. Multilevel Binary Morphological Analysis

### 2.3 Histogram Based Features
1. Histogram
2. Multi-region histogram
3. Correlogram

### 2.4 Multi-scale Features
1. Fractal Dimension Texture Analysis (FDTA)
2. Amplitude Modulation – Frequency Modulation (AM-FM)
3. Discrete Wavelet Transform (DWT)
4. Stationary Wavelet Transform (SWT)
5. Wavelet Packets (WP)
6. Gabor Transform (GT)

### 2.5 Other Features
1. Zernikes’ Moments
2. Hu’s Moments
3. Threshold Adjacency Matrix (TAS)
4. Histogram of Oriented Gradients (HOG)

## 3. How to use each feature set
For the following sections, assume
* _f_ is a grayscale image as a numpy ndarray, 
* _mask_ is an image as a numpy ndarray but with 2 values: 0 (zero) and 1 (one) with 1 indicating the region-of-interest (ROI), where the features shall be calculated (values outside ROI are ignored),
* _perimeter_ is like _mask_ but indicates the perimeter of the ROI. 
The demo has an analytic way on how to create _mask_ and _perimeter_, given a set of coordinates.  

<p align="center">
  <img width="375" height="130" src="https://github.com/giakou4/features/blob/main/demo/data/f.png">
</p>
<p align="center">
  <img width="375" height="130" src="https://github.com/giakou4/features/blob/main/demo/data/mask.png">
</p>
<p align="center">
  <img width="375" height="130" src="https://github.com/giakou4/features/blob/main/demo/data/perimeter.png">
</p>

### 3.1 Textural Features
#### 3.1.1 First Order Statistics/Statistical Features (FOS/SF)
```python
features, labels = fos(f, mask)
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.

    Returns
    -------
    features : numpy ndarray
        Feature set as defined in theory.
    labels : list
        Labels of features.
    '''
```
#### 3.1.2 Gray Level Co-occurence Matrix (GLCM/SGLDM)
```python
features_mean, features_range, labels_mean, labels_range = glcm_features(f, ignore_zeros)
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    ignore_zeros : int, optional
        Ignore zeros in image f. The default is True.

    Returns
    -------
    features_mean : numpy ndarray
        Haralick's features, mean
    features_range : numpy ndarray
        Haralick's features, same as before but range
    labels_mean : list
        Labels of features_mean.
    labels_range: list
        Labels of features_range.
    '''
```
#### 3.1.3 Gray Level Difference Statistics (GLDS)
```python
features, labels = glds_features(f, mask, Dx, Dy)
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else.
    Dx : int, optional
        Array with X-coordinates of vectors denoting orientation. The default
        is [0,1,1,1].
    Dy : int, optional
        Array with Y-coordinates of vectors denoting orientation. The default
        is [1,1,0,-1].
        
    Returns
    -------
    features : numpy ndarray
        Feature set as defined in theory.
    labels : list
        Labels of features.
    '''
```
#### 3.1.4 Neighborhood Gray Tone Difference Matrix (NGTDM)
```python
features, labels = ngtdm_features(f, mask, d)
    '''  
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    d : int, optional
        Distance for NGTDM. Default is 1.

    Returns
    -------
    features : numpy ndarray
        Feature set as defined in theory.
    labels : list
        Labels of features.
    '''
```
#### 3.1.5 Statistical Feature Matrix (SFM)
```python
features, labels = sfm_features(f, mask, Lr, Lc)
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
        Feature set as defined in theory.
    labels : list
        Labels of features.
    '''
```
#### 3.1.6 Law's Texture Energy Measures (LTE/TEM)
```python
features, labels = lte_measures(f, mask, l)
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    l : int, optional
        Law's mask size. The default is 7.

    Returns
    -------
    features : numpy ndarray
        Feature set as defined in theory.
    labels : list
        Labels of features.
    '''
```
#### 3.1.7 Fractal Dimension Texture Analysis (FDTA)
```python
h, labels = fdta(f, mask, s)
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    s : int, optional
        max resolution to calculate Hurst coefficients. The default is 3

    Returns
    -------
    h : numpy ndarray
        Hurst coefficients.
    labels : list
        Labels of h.
    '''
```
#### 3.1.8 Gray Level Run Length Matrix (GLRLM)
```python
features, labels = glrlm_features(f, mask, Ng)
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    Ng : int, optional
        Image number of gray values. The default is 256.

    Returns
    -------
    features : numpy ndarray
        Feature set as defined in theory.
    labels : list
        Labels of features.
    '''
```
#### 3.1.9 Fourier Power Spectrum (FPS)
```python
features, labels = fps(f, mask)
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.

    Returns
    -------
    features : numpy ndarray
        Feature set as defined in theory.
    labels : list
        Labels of features.
    '''
```
#### 3.1.10 Shape Parameters
```python
features, labels = shape_parameters(f, mask, perimeter, pixels_per_mm2)
    ''' 
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    perimeter : numpy ndarray
         Image N1 x N2 with 1 if pixels belongs to perimeter of ROI, 0 else.
    pixels_per_mm2 : int, optional
        Density of image f. The default is 1.

    Returns
    -------
    features : numpy ndarray
        Feature set as defined in theory.
    labels : list
        Labels of features.
    '''
```
#### 3.1.11 Gray Level Size Zone Matrix (GLSZM)
```python
features, labels = glszm_features(f, mask)
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else.

    Returns
    -------
    features : numpy ndarray
        Feature set as defined in theory.
    labels : list
        Labels of features.
    '''
```
#### 3.1.12 Higher Order Spectra (HOS)
```python
features, labels = hos_features(f, th)
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    th : list, optional
        Angle to calculate Radon Transform. The default is [135,140].

    Returns
    -------
    features : numpy ndarray
        Entropy of bispectrum of radeon transform of image for each angle in theta.
    labels : list
        Labels of features.
    '''
```
#### 3.1.13 Local Binary Pattern (LPB)
```python
features, labels = lbp_features(f, mask, P, R)
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    P : list, optional
        Number of points in neighborhood. The default is [8,16,24].
    R : list, optional
        Radius/Radii. The default is [1,2,3].

    Returns
    -------
    features : numpy ndarray
        Energy and entropy of LBP image (2 x 1).
    labels : list
        Labels of features.
    '''
```

### 3.2 Morphological Features
#### 3.2.1 Gray-scale Morphological Analysis
```python
pdf, cdf = grayscale_morphology_features(f, N)
    ''' 
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    N : np.array, optional
        Maximum number of scales. The default is 30.

    Returns
    -------
    pdf : numpy ndarray
        Probability density function (pdf) of pattern spectrum.
    cdf : numpy ndarray
        Cumulative density function (cdf) of pattern spectrum.
    '''
```
#### 3.2.2 Multilevel Binary Morphological Analysis
```python
pdf_L, pdf_M, pdf_H, cdf_L, cdf_M, cdf_H = multilevel_binary_morphology_features(img, mask, N, thresholds):
    ''' 
    Parameters
    ----------
    img : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    N : np.array, optional
        Maximum number of scales. The default is 30.
    thresholds: list, optional
        Thresholds to get the 3 binary images. The default is [25, 50].

    Returns
    -------
    pdf_L : numpy ndarray
        Probability density function (pdf) of pattern spectrum for image L.
    pdf_M : numpy ndarray
        Probability density function (pdf) of pattern spectrum for image M.
    pdf_H : numpy ndarray
        Probability density function (pdf) of pattern spectrum for image H.
    cdf_L : numpy ndarray
        Cumulative density function (cdf) of pattern spectrum for image L.
    cdf_M : numpy ndarray
        Cumulative density function (cdf) of pattern spectrum for image M.
    cdf_H : numpy ndarray
        Cumulative density function (cdf) of pattern spectrum for image H.
    '''   
```

### 3.3 Histogram Based Features
#### 3.3.1 Histogram
```python
H, labels = histogram(f, mask, bins)
    ''' 
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    bins : int, optional
         Bins for histogram. The default is 32.


    Returns
    -------
    H : numpy ndarray
        Histogram of image f for 256 gray levels.
    labels : list
        Labels of features, which are the bins' number.
    '''
```
#### 3.3.2 Multi-region histogram
```python
features, labels = multiregion_histogram(f, mask, bins, num_eros, square_size)
    ''' 
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    bins : int, optional
        Bins for histogram. Default is 32.
    num_eros : int, optional
        Times of erosion to be performed. Default is 3.
    square_size : int, optional
        Kernel where erosion is performed, here a squared. Default square's
        size is 3.
        
    Returns
    -------
    features : numpy ndarray
        Histogram of f and f after erosions as a vector e.g. [32 x num_eros].
    labels : list
        Labels of features.
    '''
```
#### 3.3.3 Correlogram
```python
Hd, Ht, labels = correlogram(f, mask, bins_digitize, bins_hist, flatten)
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
```

### 3.4 Multi-scale Features
#### 3.4.1 Fractal Dimension Texture Analysis (FDTA)
```python
h, labels = fdta(f, mask, s)
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    s : int, optional
        max resolution to calculate Hurst coefficients. The default is 3

    Returns
    -------
    h : numpy ndarray
        Hurst coefficients.
    labels : list
        Labels of h.
    '''
```
#### 3.4.2 Amplitude Modulation – Frequency Modulation (AM-FM)
```python
features, labels = amfm_features(f, bins)
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    bins: int, optional
        Bins for the calculated histogram. The default is 32.

    Returns
    -------
    features : numpy ndarray
        Histogram of IA, IP, IFx, IFy as a concatenated vector.
    labels : list
        Labels of features.
    '''
```
#### 3.4.3 Discrete Wavelet Transform (DWT)
```python
features, labels = dwt_features(f, mask, wavelet, levels)
    ''' 
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    wavelet : str, optional
         Filter to be used. Check pywt for filter families. The default is 'bior3.3'
    levels : int, optional
        Levels of decomposition. Default is 3.

    Returns
    -------
    features : numpy ndarray
        Mean and std of each detail image. Appromimation images are ignored.
    labels : list
        Labels of features.
    '''
```
#### 3.4.4 Stationary Wavelet Transform (SWT)
```python
features, labels = swt_features(f, mask, wavelet, levels)
    ''' 
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    wavelet : str, optional
         Filter to be used. Check pywt for filter families. The default is 'bior3.3'
    levels : int, optional
        Levels of decomposition. Default is 3.

    Returns
    -------
    features : numpy ndarray
        Mean and std of each detail image. Appromimation images are ignored.
    labels : list
        Labels of features.
    '''
```
#### 3.4.5 Wavelet Packets (WP)
```python
features, labels = wp_features(f, mask, wavelet, maxlevel) 
    ''' 
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    wavelet : str, optional
         Filter to be used. Check pywt for filter families. The default is 'cof1'
    maxlevel : int, optional
        Levels of decomposition. Default is 3.

    Returns
    -------
    features : numpy ndarray
        Mean and std of each detail image. Appromimation images are ignored.
    labels : list
        Labels of features.
    '''
```
#### 3.4.6 Gabor Transform (GT)
```python
features, labels = gt_features(f, mask, deg, freq)
    ''' 
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    deg: int, optinal
        Quantized degrees. The default is 4 (0, 45, 90, 135 degrees)
    freq: list, optional
        frequency of the gabor kernel. The default is [0.05, 0.4]

    Returns
    -------
    features : numpy ndarray
        Mean and std for the resulted image: (f o gabor_filter)(x,y)
        
    labels : list
        Labels of features.
    '''  
```

### 3.5 Other Features
#### 3.5.1 Zernikes’ Moments
```python
features, labels = zernikes_moments(f, radius)
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    radius : int, optional
        Radius to calculate Zernikes moments. The default is 9.

    Returns
    -------
    features : numpy ndarray
        Zernikes' moments.
    labels : list
        Labels of features.
    '''
```
#### 3.5.2 Hu’s Moments
```python
features, labels = hu_moments(f)
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.

    Returns
    -------
    features : numpy ndarray
        Hu's moments.
    labels : list
        Labels of features.
    '''
```
#### 3.5.3 Threshold Adjacency Matrix (TAS)
```python
features, labels = tas_features(f)
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
```
#### 3.5.4 Histogram of Oriented Gradients (HOG)
```python
fd, labels = hog_features(f, ppc, cpb)
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
```

## 4. Citation
In Bibtex format:

>   @misc{Giakoumoglou2021,  
>     author = {Nikolaos Giakoumoglou},  
>     title = {PyFeats: Open source software for image feature extraction},  
>     year = {2021},  
>     publisher = {GitHub},  
>     journal = {GitHub repository},  
>     howpublished = {\url{https://github.com/giakou4/pyfeats}},  
>   }  


## 5. Support
Reach out to me:
- [giakou4's email](mailto:ngiakoumoglou@hotmail.com "ngiakoumoglou@hotmail.com")

## 6. Python Libraries Cited:
* Bradski, G. (2000). The OpenCV Library. Dr. Dobb&#x27;s Journal of Software Tools.
* Coelho, L.P. 2013. Mahotas: Open source software for scriptable computer vision. Journal of Open Research Software 1(1):e3, DOI: http://dx.doi.org/10.5334/jors.ac
* Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science &amp; Engineering, 9(3), 90–95.
* Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 0.1038/s41586-020-2649-2. (Publisher link).
* Gregory R. Lee, Ralf Gommers, Filip Wasilewski, Kai Wohlfahrt, Aaron O’Leary (2019). PyWavelets: A Python package for wavelet analysis. Journal of Open Source Software, 4(36), 1237, https://doi.org/10.21105/joss.01237.
* Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., … SciPy 1.0 Contributors. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17, 261–272. https://doi.org/10.1038/s41592-019-0686-2
* Van der Walt, S., Sch"onberger, Johannes L, Nunez-Iglesias, J., Boulogne, Franccois, Warner, J. D., Yager, N., … Yu, T. (2014). scikit-image: image processing in Python. PeerJ, 2, e453.

## 7. Features Theory Citation:
* Acharya U, R., Chua, C. K., Ng, E. Y., Yu, W., & Chee, C. (2008, 12 01). Application of Higher Order Spectra for the Identification of Diabetes Retinopathy Stages. Journal of Medical Systems, 32, 481-488. doi:10.1007/s10916-008-9154-8
* Acharya, U. R., Chua, K., Lim, T.-C., Tay, D., & Suri, J. (2009, 12). Automatic identification of epileptic EEG signals using nonlinear parameters. Journal of Mechanics in Medicine and Biology, 9, 539-553. doi:10.1142/S0219519409003152
* Amadasun, M., & King, R. (1989). Textural features corresponding to textural properties. IEEE Trans. Syst. Man Cybern., 19, 1264-1274.
* Chua, K., Chandran, V., Acharya, U. R., & Lim, C. (2009, 6). Automatic identification of epileptic electroencephalography signals using higher-order spectra. Proceedings of the Institution of Mechanical Engineers. Part H, Journal of engineering in medicine, 223, 485-95. doi:10.1243/09544119JEIM484
* Chua, K., Chandran, V., Acharya, U. R., & Lim, C. (2011, 12). Application of Higher Order Spectra to Identify Epileptic EEG. Journal of medical systems, 35, 1563-71. doi:10.1007/s10916-010-9433-z
* Galloway, M. M. (1975). Texture analysis using gray level run lengths. Computer Graphics and Image Processing, 4, 172-179. doi:https://doi.org/10.1016/S0146-664X(75)80008-6
* Haralick, R., Shanmugam, K., & Dinstein, I. (1973, 1). Textural Features for Image Classification. IEEE Trans Syst Man Cybern, SMC-3, 610-621.
* Hu, M. (1962). Visual pattern recognition by moment invariants. IRE Trans. Inf. Theory, 8, 179-187.
* Laws, K. (1980). Rapid texture identification.
* Lee, G. R., Gommers, R., Waselewski, F., Wohlfahrt, K., O&amp, A., #8217, & Leary. (2019). PyWavelets: A Python package for wavelet analysis. Journal of Open Source Software, 4, 1237. doi:10.21105/joss.01237
* Liu, M., He, Y., & Ye, B. (2007). Image Zernike moments shape feature evaluation based on image reconstruction. Geo-spatial Information Science, 10, 191-195. doi:10.1007/s11806-007-0060-x
* Mandelbrot, B. (1977). Fractal Geometry of Nature.
* Maragos, P. (1989, 8). Pattern spectrum and multiscale shape representation. IEEE Trans Pattern Anal Mach Intell. Pattern Analysis and Machine Intelligence, IEEE Transactions on, 11, 701-716. doi:10.1109/34.192465
* Maragos, P., & Ziff, R. (1990, 6). Threshold Superposition in Morphological Image Analysis Systems. Pattern Analysis and Machine Intelligence, IEEE Transactions on, 12, 498-504. doi:10.1109/34.55110
* Murray Herrera, V. M. (2009). AM-FM methods for image and video processing. Retrieved from https://digitalrepository.unm.edu/
* Ojala, T., Pietikäinen, M., & Harwood, D. (1996). A comparative study of texture measures with classification based on featured distributions. Pattern Recognit., 29, 51-59.
* Ojala, T., Pietikäinen, M., & Maenpaa, T. (2002, 8). Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns. Pattern Analysis and Machine Intelligence, IEEE Transactions on, 24, 971-987. doi:10.1109/TPAMI.2002.1017623
* Teague, M. R. (1980, 8). Image analysis via the general theory of moments∗. J. Opt. Soc. Am., 70, 920–930. doi:10.1364/JOSA.70.000920
* Thibault, G., Fertil, B., Navarro, C., Pereira, S., Cau, P., Lévy, N., . . . Mari, J. (2009). Texture indexes and gray level size zone matrix. Application to cell nuclei classification.
* Toet, A. (1990). A hierarchical morphological image decomposition. Pattern Recognition Letters, 11, 267-274. doi:https://doi.org/10.1016/0167-8655(90)90065-A
* Tsiaparas, N., Golemati, S., Andreadis, I., Stoitsis, J. S., Valavanis, I., & Nikita, K. S. (2011, 1). Comparison of Multiresolution Features for Texture Classification of Carotid Atherosclerosis From B-Mode Ultrasound. IEEE Transactions on Information Technology in Biomedicine, 15, 130–137. doi:10.1109/titb.2010.2091511
* Weszka, J. S., Dyer, C. R., & Rosenfeld, A. (1976). A Comparative Study of Texture Measures for Terrain Classification. IEEE Transactions on Systems, Man, and Cybernetics, SMC-6, 269-285. doi:10.1109/TSMC.1976.5408777
* Wu, C.-M., & Chen, Y.-C. (1992). Statistical feature matrix for texture analysis. CVGIP: Graphical Models and Image Processing, 54, 407-419. doi:https://doi.org/10.1016/1049-9652(92)90025-S
* Wu, C.-M., Chen, Y.-C., & Hsieh, K.-S. (1992). Texture features for classification of ultrasonic liver images. IEEE Transactions on Medical Imaging, 11, 141-152. doi:10.1109/42.141636

