<p align="center">
  <img src="https://github.com/giakou4/features/blob/main/demo/data/PyFeats Logo v2.png">
</p>

# PyFeats

Open-source software for image feature extraction

[![DOI](https://zenodo.org/badge/371957337.svg)](https://zenodo.org/badge/latestdoi/371957337)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/giakou4/pyfeats/LICENSE)
[![version](https://img.shields.io/pypi/v/pyfeats)](https://pypi.org/project/pyfeats/)
[![Downloads](https://pepy.tech/badge/pyfeats)](https://pepy.tech/project/pyfeats)
![stars](https://img.shields.io/github/stars/giakou4/pyfeats.svg)
![issues-open](https://img.shields.io/github/issues/giakou4/pyfeats.svg)
![issues-closed](https://img.shields.io/github/issues-closed/giakou4/pyfeats.svg)
![size](https://img.shields.io/github/languages/code-size/giakou4/pyfeats)

[comment]:[![Downloads](https://pepy.tech/badge/pyfeats/month)](https://pepy.tech/project/pyfeats)
[comment]:[![Downloads](https://pepy.tech/badge/pyfeats/week)](https://pepy.tech/project/pyfeats)
[comment]:[![PyPi](https://badgen.net/badge/icon/pypi?icon=pypi&label)](https://pypi.org/project/pyfeats/)
[comment]:![forks](https://img.shields.io/github/forks/giakou4/pyfeats.svg)

## Updates

* 13/06/2023 - Stargazers added on README.md (using [star-history](https://star-history.com/))
* 10/06/2023 - New logo
* 29/05/2023 - Post on [medium](https://medium.com/@giakoumoglou4/pyfeats-open-source-software-for-image-feature-extraction-47f43bb33563)

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

The first step for every machine learning algorithm in a computer vision problem is to extract features from the given images. Feature extraction is a critical step in any pattern classification system. In order for the pattern recognition process to be tractable, it is necessary to convert patterns into features, which are condensed representations of the patterns, containing only salient information. Features contain the characteristics of a pattern in a comparable form making the pattern classification possible. Feature extraction can be accomplished manually or automatically. 

Manual feature extraction requires identifying and describing the features that are relevant for a given problem and implementing a way to extract those features. In many situations, having a good understanding of the background or domain can help make informed decisions as to which features could be useful. Over decades of research, engineers and scientists have developed feature extraction methods for images. The features inlcuded in here are: Textural Features, Morphological Features, Histogram-based Features, Multi-scale Features, and Moments.

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

Note than an analytical description of how each feature is calculated lies in ```pyfeats.pdf```.

### 3.1 Textural Features
#### 3.1.1 First Order Statistics/Statistical Features (FOS/SF)
First Order Statistics (FOS) are calculated from the histogram of the image which is the empirical probability density function for single pixels. The FOS features are the following:  1) mean, 2) standard deviation, 3) median, 4) mode, 5) skewnewss, 6) kurtosis, 7) energy, 8) entropy, 9) minimal gray level, 10) maximal gray leve, 11) coefficient of variation, 12,13,14,15) percentiles (10, 25, 50, 75, 90) and 16) histogram width.
```python
features, labels = pyfeats.fos(f, mask)
```
#### 3.1.2 Gray Level Co-occurence Matrix (GLCM/SGLDM)
The Gray Level Co-occurrence Matrix (GLCM) as proposed by Haralick are based on the estimation of the second-order joint conditional probability density functions. The GLGLCM features are the following:  1) angular second moment, 2) contrast, 3) correlation, 4) sum of squares: variance, 5) inverse difference moment, 6) sum average, 7) sum variance, 8) sum entropy, 9) entropy, 10) difference variance, 11) difference entropy, 12,13) information measures of correlation. For each feature, the mean values and the range of values are computed, and are used as two different features sets.
```python
features_mean, features_range, labels_mean, labels_range = pyfeats.glcm_features(f, ignore_zeros=True)
```
#### 3.1.3 Gray Level Difference Statistics (GLDS)
The Gray Level Difference Statistics (GLDS) algorithm uses first order statistics of local property values based on absolute differences between pairs of gray levels or of average gray levels in order to extract texture measures. The GLDS features are the following:  1) homogeneity, 2) contrast, 3) energy, 4) entropy, 5) mean.
```python
features, labels = pyfeats.glds_features(f, mask, Dx=[0,1,1,1], Dy=[1,1,0,-1])
```
#### 3.1.4 Neighborhood Gray Tone Difference Matrix (NGTDM)
Neighbourhood Gray Tone Difference Matrix (NDTDM) corresponds to visual properties of texture. The NGTDM features are the following:  1) coarseness, 2) contrast, 3) busyness, 4) complexity, 5) strength.
```python
features, labels = pyfeats.ngtdm_features(f, mask, d=1)
```
#### 3.1.5 Statistical Feature Matrix (SFM)
The Statistical Feature Matrix measures the statistical properties of pixel pairs at several distances within an image which are used for statistical analysis. The SFM features are the following: 1) coarseness, 2) contrast, 3) periodicity, 4) roughness.
```python
features, labels = pyfeats.sfm_features(f, mask, Lr=4, Lc=4)
```
#### 3.1.6 Law's Texture Energy Measures (LTE/TEM)
Law’s texture Energy Measures, are derived from three simple vectors of length 3. If these vectors are convolved with themselves, new vectors of length 5 are obtained. By further self-convolution, new vectors of length 7 are obtained. If the column vectors of length l are multiplied by row vectors of the same length, Laws l×l masks are obtained. In order to extract texture features from an image, these masks are convoluted with the image, and the statistics (e.g., energy) of the resulting image are used to describe texture: 1) texture energy from LL kernel, 2) texture energy from EE kernel, 3) texture energy from SS kernel, 4) average texture energy from LE and EL kernels, 5) average texture energy from ES and SE kernels, 6) average texture energy from LS and SL kernels.
```python
features, labels = pyfeats.lte_measures(f, mask, l=7)
```
#### 3.1.7 Fractal Dimension Texture Analysis (FDTA)
Fractal Dimension Texture Analysis (FDTA)  is based on the Fractional Brownian Motion (FBM) Model. The FBM model is used to describe the roughness of nature surfaces. It regards naturally occurring surfaces as the end result of random walks. Such random walks are basic physical processes in our universe. One of the most important parameters to represent a fractal surface is the fractal dimension. A simpler method is to estimate the H parameter (Hurst coefficient). If the image is seen under different resolutions, then the multiresolution fractal (MF) feature vector is obtained.
```python
h, labels = pyfeats.fdta(f, mask, s=3)
```
#### 3.1.8 Gray Level Run Length Matrix (GLRLM)
 A gray level run is a set of consecutive, collinear picture points having the same gray level value. The length of the run is the number of picture points in the run. The GLRLM features are the following: 1) short run emphasis, 2) long run emphasis, 3) gray level non-uniformity, 4) run length non-uniformity, 5) run percentage, 6) low gray level run emphasis, 7) high gray level run emphasis, 8) short low gray level emphasis, 9) short run high gray level emphasis, 10) long run low gray level emphasis, 11) long run high level emphasis.
```python
features, labels = pyfeats.glrlm_features(f, mask, Ng=256)
```
#### 3.1.9 Fourier Power Spectrum (FPS)
For digital pictures, instead of the continuous Fourier transform, one uses the discrete transform. The standard set of texture features based on a ring-shaped samples of the discrete Fourier power spectrum are of the form. Similarly, the features based on a wedge-shaped samples are of the form.
The FPS features are the following: 1) radial sum, 2) angular sum
```python
features, labels = pyfeats.fps(f, mask)
```
#### 3.1.10 Shape Parameters
Shape parameters consists of the following features: 1) x-coordinate maximum length, 2) y-coordinate maximum length, 3) area, 4) perimeter, 5) perimeter2/area
```python
features, labels = pyfeats.shape_parameters(f, mask, perimeter, pixels_per_mm2=1)
```
#### 3.1.11 Gray Level Size Zone Matrix (GLSZM)
Gray Level Size Zone Matrix (GLSZM) quantifies gray level zones in an image. A gray level zone is defined as the number of connected voxels that share the same gray level intensity. A voxel is considered connected if the distance is 1 according to the infinity norm (26-connected region in a 3D, 8-connected region in 2D). The GLSZM features are the following: 1) small zone emphasis, 2) large zone emphasis, 3) gray level non-uniformity, 4) zone-size non-uniformity, 5) zone percentage, 6) low gray level zone emphasis, 7) high gray level zone emphasis, 8) small zone low gray level emphasis, 9) small zone high gray level emphasis, 10) large zone low gray level emphasis, 11) large zone high gray level emphasis, 12 gray level variance, 13) zone-size variance, 14) zone-size entropy.
```python
features, labels = pyfeats.glszm_features(f, mask, connectivity=1)
```
#### 3.1.12 Higher Order Spectra (HOS)
Radon transform transforms two dimensional images with lines into a domain of possible line parameters, where each line in the image will give a peak positioned at the corresponding line parameters. Hence, the lines of the images are transformed into the points in the Radon domain. High Order Spectra (HOS) are spectral components of higher moments. The bispectrum, of a signal is the Fourier transform (FT) of the third order correlation of the signal (also known as the third order cumulant function). The bispectrum, is a complex-valued function of two frequencies. The bispectrum which is the product of three Fourier coefficients, exhibits symmetry and was computed in the non-redundant region. The extracted feature is the entropy 1.
```python
features, labels = pyfeats.hos_features(f, th=[135,140])
```
#### 3.1.13 Local Binary Pattern (LPB)
Local Binary Pattern (LBP), a robust and efficient texture descriptor, was first presented by Ojala. The LBP feature vector, in its simplest form, is determined using the following method: A circular neighbourhood is considered around a pixel. P points are chosen on the circumference of the circle with radius R such that they are all equidistant from the centre pixel. . These P points are converted into a circular bit-stream of 0s and 1s according to whether the gray value of the pixel is less than or greater than the gray value of the centre pixel. Ojala et al. (2002) introduced the concept of uniformity in texture analysis. The uniform fundamental patterns have a uniform circular structure that contains very few spatial transitions U (number of spatial bitwise 0/1 transitions). In this work, a rotation invariant measure using uniformity measure U was calculated. Only patterns with U less than 2 were assigned the LBP code i.e., if the number of bit transitions in the circular bit-stream is less than or equal to 2, the centre pixel was labelled as uniform. Energy and entropy of the LBP image, constructed over different scales are used as feature descriptors.
```python
features, labels = pyfeats.lbp_features(f, mask, P=[8,16,24], R=[1,2,3])
```

### 3.2 Morphological Features
#### 3.2.1 Gray-scale Morphological Analysis
In multilevel binary morphological analysis different components are extracted and investigated for their geometric properties. Three binary images are generated by thresholding. Here, binary image outputs are represented as sets of image coordinates where image intensity meets the threshold criteria. Overall, this multilevel decomposition is closely related to a three-level quantization of the original image intensity.  For each binary image the pattern spectrum is calculated. The Grayscale Morphological Features are the following: 1) mean cumulative distribution functions (CDF) and 2) mean probability density functions (PDF) of the pattern spectra  using the cross $"+"$ as a structural element of the grayscale image.
```python
pdf, cdf = pyfeats.grayscale_morphology_features(f, N)
```
#### 3.2.2 Multilevel Binary Morphological Analysis
Same as above but with grayscale image. The difference lies in the calculation of the pattern spectrum.
```python
pdf_L, pdf_M, pdf_H, cdf_L, cdf_M, cdf_H = pyfeats.multilevel_binary_morphology_features(f, mask, N=30, thresholds=[25, 50]):
```

### 3.3 Histogram Based Features
#### 3.3.1 Histogram
Histogram: The grey level histogram of the ROI of the image.
```python
H, labels = pyfeats.histogram(f, mask, bins=32)
```
#### 3.3.2 Multi-region histogram
A number of equidistant ROIs are identified by eroding the image outline by a factor based on the image size. The histogram is computed for each one of the regions as described above
```python
features, labels = pyfeats.multiregion_histogram(f, mask, bins, num_eros=3, square_size=3)
```
#### 3.3.3 Correlogram
Correlograms are histograms, which measure not only statistics about the features of the image, but also consider the spatial distribution of these features. In this work two correlograms are implemented for the ROI of the  image:

* based on the distance of the distribution of the pixels’ gray level values from the centre of the image, and
* based on their angle of distribution.

For each pixel the distance and the angle from the image centre is calculated and for all pixels with the same distance or angle their histograms are computed. In order to make the comparison between images of different sizes feasible, the distance correlograms is normalized. The angle of the correlogram is allowed to vary among possible values starting from the left middle of the image and moving clockwise. The resulting correlograms are matrices.
```python
Hd, Ht, labels = pyfeats.correlogram(f, mask, bins_digitize=32, bins_hist=32, flatten=True)
```

### 3.4 Multi-scale Features
#### 3.4.1 Fractal Dimension Texture Analysis (FDTA)
See 3.1.7.
```python
h, labels = pyfeats.fdta(f, mask, s=3)
```
#### 3.4.2 Amplitude Modulation – Frequency Modulation (AM-FM)
We consider multi-scale Amplitude Modulation – Frequency Modulation (AM-FM) representations, under least-square approximations, for images. For each image an instantaneous amplitude (IA), an the instantaneous phase (IP) and an instantaneous frequency (IF) is calculated for a specific component. Given the input discrete image, we first apply the Hilbert transform to form a 2D extension of the 1D analytic signal. The result is processed through a collection of bandpass filters with the desired scale. Each processing block will produce the IA, the IP and the IF. The AM-FM features are the following: Histogram of the 1) low, 2) medium, 3) high and 4) dc reconstructed images.
```python
features, labels = pyfeats.amfm_features(f, bins=32)
```
#### 3.4.3 Discrete Wavelet Transform (DWT)
The Discrete Wavelet Transform (DWT) of a signal is defined as its inner product with a family of functions. For images, i.e., 2-D signals, the 2-D DWT can be used. This consists of a DWT on the rows of the image and a DWT on the columns of the resulting image. The result of each DWT is followed by down sampling on the columns and rows, respectively. The decomposition of the image yields four sub-images for every level. Each approximation sub-image is decomposed into four sub images named approximation, detail-horizontal, detail-vertical, and detail-diagonal sub-image respectively. Each detail sub-image is the result of a convolution with two half-band filters. The DWT features are the following: 1) mean and 2) standard deviation of the absolute value of detail sub-images of the DWT.
```python
features, labels = pyfeats.dwt_features(f, mask, wavelet='bior3.3', levels=3)
```
#### 3.4.4 Stationary Wavelet Transform (SWT)
The 2-D Stationary Wavelet Transform (SWT) is similar to the 2-D DWT, but no down sampling is performed. Instead, up sampling of the low-pass and high-pass filters is carried out. The SWT features are the following: 1) mean and 2) standard deviation of the absolute value of detail sub-images of the SWT.
```python
features, labels = pyfeats.swt_features(f, mask, wavelet='bior3.3', levels=3)
```
#### 3.4.5 Wavelet Packets (WP)
The 2-D Wavelet Packets (WP) decomposition is a simple modification of the 2-D DWT, which offers a richer space-frequency representation. The first level of analysis is the same as that of the 2-D DWT. The second, as well as all subsequent levels of analysis consist of decomposing every sub image, rather than only the approximation sub image, into four new sub images. The WP features are the following: 1) mean and 2) standard deviation of the absolute value of detail sub-images of the Wavelet Decomposition.
```python
features, labels = pyfeats.wp_features(f, mask, wavelet='cof1', maxlevel=3) 
```
#### 3.4.6 Gabor Transform (GT)
The Gabor Transform (GT) of an image consists in convolving that image with the Gabor function, i.e., a sinusoidal plane wave of a certain frequency and orientation modulated by a Gaussian envelope. Frequency and orientation representations of Gabor filters are similar to those of the human visual system, rendering them appropriate for texture segmentation and classification. The GT features are the following: 1) mean and 2) standard deviation of the absolute value of detail sub-images of the GT of the image.
```python
features, labels = pyfeats.gt_features(f, mask, deg=4, freq=[0.05, 0.4])
```

### 3.5 Other Features
#### 3.5.1 Zernikes’ Moments
In image processing, computer vision and related fields, an image moment is a certain particular weighted average (moment) of the image pixels' intensities, or a function of such moments, usually chosen to have some attractive property or interpretation. 
Zernikes’ moment is a kind of orthogonal complex moments and its kernel is a set of Zernike complete orthogonal polynomials defined over the interior of the unit disc in the polar coordinates space. Zernike's Moments are: 1-25) orthogonal moments invariants with respect to translation.
```python
features, labels = pyfeats.zernikes_moments(f, radius=9)
```
#### 3.5.2 Hu’s Moments
In image processing, computer vision and related fields, an image moment is a certain particular weighted average (moment) of the image pixels' intensities, or a function of such moments, usually chosen to have some attractive property or interpretation. 
Hu’s Moments are: 1-7) moments invariants with respect to translation, scale, and rotation.
```python
features, labels = pyfeats.hu_moments(f)
```
#### 3.5.3 Threshold Adjacency Matrix (TAS)
```python
features, labels = pyfeats.tas_features(f)
```
#### 3.5.4 Histogram of Oriented Gradients (HOG)
```python
fd, labels = pyfeats.hog_features(f, ppc=8, cpb=3)
```

## 4. Citation
In Bibtex format:
```bibtex
@misc{pyfeats2021giakoumoglou,  
     author = {Nikolaos Giakoumoglou},  
     title = {PyFeats: Open-source software for image feature extraction},  
     year = {2021},  
     publisher = {GitHub},  
     journal = {GitHub repository},  
     howpublished = {\url{https://github.com/giakou4/pyfeats}},  
   }  
```

## 5. Support
Reach out to me:
- [giakou4's email](mailto:giakou4@gmail.com "giakou4@gmail.com")

## 6. Star History

[![Star History Chart](https://api.star-history.com/svg?repos=giakoumoglou/pyfeats&type=Date)](https://star-history.com/#giakoumoglou/pyfeats&Date)

## 7. Python Libraries Cited:
* Bradski, G. (2000). The OpenCV Library. Dr. Dobb&#x27;s Journal of Software Tools.
* Coelho, L.P. 2013. Mahotas: Open source software for scriptable computer vision. Journal of Open Research Software 1(1):e3, DOI: http://dx.doi.org/10.5334/jors.ac
* Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science &amp; Engineering, 9(3), 90–95.
* Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 0.1038/s41586-020-2649-2. (Publisher link).
* Gregory R. Lee, Ralf Gommers, Filip Wasilewski, Kai Wohlfahrt, Aaron O’Leary (2019). PyWavelets: A Python package for wavelet analysis. Journal of Open Source Software, 4(36), 1237, https://doi.org/10.21105/joss.01237.
* Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., … SciPy 1.0 Contributors. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17, 261–272. https://doi.org/10.1038/s41592-019-0686-2
* Van der Walt, S., Sch"onberger, Johannes L, Nunez-Iglesias, J., Boulogne, Franccois, Warner, J. D., Yager, N., … Yu, T. (2014). scikit-image: image processing in Python. PeerJ, 2, e453.

## 🧪 New Example: Streamlit Web App for Shape Classification

A simple Streamlit-based web app to classify basic geometric shapes using Zernike Moments and SVM.

### 🔹 Features:
- Upload shape image (JPG/PNG)
- Predict shape: Circle, Square, Triangle
- Confidence score and bar chart of class probabilities
- Uses `pyfeats.zernikes_moments` for feature extraction

### 🚀 How to Run

1. Train the model first with:


## 8. Features Theory Citation:
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

