import pywt
from .dwt import dwt_features
from .swt import swt_features
from .wp import wp_features
from .gt import gt_features
from .amfm import amfm_features
from .fdta import fdta

__all__ = ['fdta',
           'dwt_features',
           'swt_features', 
           'wp_features', 
           'gt_features', 
           'amfm_features']

__wavelets__ = pywt.wavelist(family=None, kind ='all')