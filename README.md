# Image Feature Extraction in ROI (Python)
A collection of python functions for feature extraction. The features are calculated inside a Region of Interest (ROI) and not for the whole image: the image is actually a polygon! More and more features will be added. Please feel free to point out any mistakes or improvemets. The aim is to create a library for image feature extraction. Message me for more details.

## Features

### A. Textural Features
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

### B. Morphological Features
1. Grayscale Morphological Analysis
2. Multilevel Binary Morphological Analysis

### C. Histogram Based Features
1. Histogram
2. Multi-region histogram
3. Correlogram

### D. Multi-scale Features
1. Fractal Dimension Texture Analysis (FDTA)
2. Amplitude Modulation – Frequency Modulation (AM-FM)
3. Discrete Wavelet Transform (DWT)
4. Stationary Wavelet Transform (SWT)
5. Wavelet Packets (WP)
6. Gabor Transform (GT)

### E. Other Features
1. Zernikes’ Moments
2. Hu’s Moments
3. Threshold Adjacency Matrix (TAS)
4. Histogram of Oriented Gradients (HOG)

## Use
Download the folder features, add to path and call
```python
from features import *
```

## Support
Reach out to me:
- [giakou4's email](mailto:giakonick98@gmail.com "giakonick98@gmail.com")

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/giakou4/Features/LICENSE)

## Python Libraries Cited:
* Bradski, G. (2000). The OpenCV Library. Dr. Dobb&#x27;s Journal of Software Tools.
* Coelho, L.P. 2013. Mahotas: Open source software for scriptable computer vision. Journal of Open Research Software 1(1):e3, DOI: http://dx.doi.org/10.5334/jors.ac
* Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science &amp; Engineering, 9(3), 90–95.
* Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 0.1038/s41586-020-2649-2. (Publisher link).
* Gregory R. Lee, Ralf Gommers, Filip Wasilewski, Kai Wohlfahrt, Aaron O’Leary (2019). PyWavelets: A Python package for wavelet analysis. Journal of Open Source Software, 4(36), 1237, https://doi.org/10.21105/joss.01237.
* Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., … SciPy 1.0 Contributors. (2020). SciPy 1.0: Fundamental Algorithms for Scientific * * * Computing in Python. Nature Methods, 17, 261–272. https://doi.org/10.1038/s41592-019-0686-2
* Van der Walt, S., Sch"onberger, Johannes L, Nunez-Iglesias, J., Boulogne, Franccois, Warner, J. D., Yager, N., … Yu, T. (2014). scikit-image: image processing in Python. PeerJ, 2, e453.

## Features Theory Citation:
* R. Haralick, K. Shanmugam and I. Dinstein, “Textural Features for Image Classification,” IEEE Trans Syst Man Cybern, Vols. SMC-3, pp. 610-621, 1 1973.
* J. S. Weszka, C. R. Dyer and A. Rosenfeld, “A Comparative Study of Texture Measures for Terrain Classification,” IEEE Transactions on Systems, Man, and Cybernetics, Vols. SMC-6, pp. 269-285, 1976.
* M. Amadasun and R. King, “Textural features corresponding to textural properties,” IEEE Trans. Syst. Man Cybern., vol. 19, pp. 1264-1274, 1989.
* C.-M. Wu and Y.-C. Chen, “Statistical feature matrix for texture analysis,” CVGIP: Graphical Models and Image Processing, vol. 54, pp. 407-419, 1992.
* K. Laws, “Rapid texture identification,” 1980.
* C.-M. Wu, Y.-C. Chen and K.-S. Hsieh, “Texture features for classification of ultrasonic liver images,” IEEE Transactions on Medical Imaging, vol. 11, pp. 141-152, 1992.
* B. Mandelbrot, “Fractal Geometry of Nature,” 1977.
* M. M. Galloway, “Texture analysis using gray level run lengths,” Computer Graphics and Image Processing, vol. 4, pp. 172-179, 1975.
* K. Chua, V. Chandran, U. R. Acharya and C. Lim, “Application of Higher Order Spectra to Identify Epileptic EEG,” Journal of medical systems, vol. 35, pp. 1563-71, 12 2011.
* K. Chua, V. Chandran, U. R. Acharya and C. Lim, “Automatic identification of epileptic electroencephalography signals using higher-order spectra,” Proceedings of the Institution of Mechanical Engineers. Part H, Journal of engineering in medicine, vol. 223, pp. 485-95, 6 2009.
* U. R. Acharya, K. Chua, T.-C. Lim, D. Tay and J. Suri, “Automatic identification of epileptic EEG signals using nonlinear parameters,” Journal of Mechanics in Medicine and Biology, vol. 9, pp. 539-553, 12 2009.
* R. Acharya U, C. K. Chua, E. Y. K. Ng, W. Yu and C. Chee, “Application of Higher Order Spectra for the Identification of Diabetes Retinopathy Stages,” Journal of Medical Systems, vol. 32, pp. 481-488, 01 12 2008.
* T. Ojala, M. Pietikäinen and D. Harwood, “A comparative study of texture measures with classification based on featured distributions,” Pattern Recognit., vol. 29, pp. 51-59, 1996.
* T. Ojala, M. Pietikäinen and T. Maenpaa, “Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns,” Pattern Analysis and Machine Intelligence, IEEE Transactions on, vol. 24, pp. 971-987, 8 2002.
* G. Thibault, B. Fertil, C. Navarro, S. Pereira, P. Cau, N. Lévy, J. Sequeira and J. Mari, “Texture indexes and gray level size zone matrix. Application to cell nuclei classification,” 2009.
* P. Maragos, “Pattern spectrum and multiscale shape representation. IEEE Trans Pattern Anal Mach Intell,” Pattern Analysis and Machine Intelligence, IEEE Transactions on, vol. 11, pp. 701-716, 8 1989.
* P. Maragos and R. Ziff, “Threshold Superposition in Morphological Image Analysis Systems.,” Pattern Analysis and Machine Intelligence, IEEE Transactions on, vol. 12, pp. 498-504, 6 1990.
* A. Toet, “A hierarchical morphological image decomposition,” Pattern Recognition Letters, vol. 11, pp. 267-274, 1990.
* N. Tsiaparas, S. Golemati, I. Andreadis, J. S. Stoitsis, I. Valavanis and K. S. Nikita, “Comparison of Multiresolution Features for Texture Classification of Carotid Atherosclerosis From B-Mode Ultrasound,” IEEE Transactions on Information Technology in Biomedicine, vol. 15, p. 130–137, 1 2011.
* G. R. Lee, R. Gommers, F. Waselewski, K. Wohlfahrt, A. O&amp, #8217 and Leary, “PyWavelets: A Python package for wavelet analysis,” Journal of Open Source Software, vol. 4, p. 1237, 2019.
* V. M. Murray Herrera, “AM-FM methods for image and video processing,” 2009. [Online]. Available: https://digitalrepository.unm.edu/.
* M. Hu, “Visual pattern recognition by moment invariants,” IRE Trans. Inf. Theory, vol. 8, pp. 179-187, 1962.
* M. R. Teague, “Image analysis via the general theory of moments∗,” J. Opt. Soc. Am., vol. 70, p. 920–930, 8 1980.
* M. Liu, Y. He and B. Ye, “Image Zernike moments shape feature evaluation based on image reconstruction,” Geo-spatial Information Science, vol. 10, pp. 191-195, 2007.
