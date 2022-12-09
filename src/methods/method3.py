import numpy as np
from skimage import exposure

from utils.img.scaler import scale

def apply_method3(img):
  ### high frequency emphasis
  v = np.fft.fftfreq(img.shape[0])
  u = np.fft.fftfreq(img.shape[1])
  vv, uu = np.meshgrid(v, u, indexing='ij')
  k1 = 0.5  
  k2 = 0.75
  D0 = 1/10
  filter_f = k1 + k2*(1 - np.exp(-np.hypot(uu, vv)**2/(2*D0**2)))
  img_f = np.fft.fft2(img)
  img_filtered_s_2 = np.fft.ifft2(img_f*filter_f).real
  ### adaptive histogram equalization
  img_filtered_s_2 = scale(img_filtered_s_2)
  img_filtered_s_equalized = exposure.equalize_adapthist(img_filtered_s_2,clip_limit=0.00015,nbins=26200)
  ### gamma
  img_filtered_s_equalized_gamma = img_filtered_s_equalized**1.2
  return img_filtered_s_equalized_gamma
