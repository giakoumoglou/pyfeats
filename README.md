# Image Feature Extraction in ROI (Python)
A collection of python functions for feature extraction. The features are calculated inside a Region of Interest (ROI) and not for the whole image: the image is actually a polygon! More and more features will be added. Please feel free to point out any mistakes or improvemets. The aim is to create a library for image feature extraction. Message me for more details.
This package was part of my thesis which was about classification of plaques extracted from ultrasound images. The example provided in the demo folder originates from it.

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
