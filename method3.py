import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy import ndimage, misc
from skimage import exposure

from utils.patterns.rectangle import get_rectangle
from utils.kernels.sobel_kernels import get_sobel_x_kernel, get_sobel_y_kernel
from utils.kernels.zeropad_kernel import get_zeropadded_kernel
from utils.kernels.gaussian_kernel import get_gaussian_kernel
from utils.file.dva_reader import read_dva
from utils.file.dva_reader import read_dva
from utils.img.scaler import scale

import cv2 as cv

### read dva
img = read_dva('./data/PATIENT_28_1.XA.0001.0001.2020.05.26.07.40.37.199459.139512372.IMA')
# np.savetxt('./outputs/img' + '.txt', img, delimiter='\t')
plt.imshow(img, cmap='gray')
# plt.show()

### high frequency emphasis
v = np.fft.fftfreq(img.shape[0])
u = np.fft.fftfreq(img.shape[1])
vv, uu = np.meshgrid(v, u, indexing='ij')
k1 = 0.5  
k2 = 0.75
### applying highfreq filter with different D0 values
# D0 = 100
# filter_f = k1 + k2*(1 - np.exp(-np.hypot(uu, vv)**2/(2*D0**2)))
# img_f = np.fft.fft2(img)
# img_filtered_s_0 = np.fft.ifft2(img_f*filter_f).real
# D0 = 10
# filter_f = k1 + k2*(1 - np.exp(-np.hypot(uu, vv)**2/(2*D0**2)))
# img_f = np.fft.fft2(img)
# img_filtered_s_1 = np.fft.ifft2(img_f*filter_f).real
D0 = 1/10
filter_f = k1 + k2*(1 - np.exp(-np.hypot(uu, vv)**2/(2*D0**2)))
img_f = np.fft.fft2(img)
img_filtered_s_2 = np.fft.ifft2(img_f*filter_f).real
# D0 = 1/100
# filter_f = k1 + k2*(1 - np.exp(-np.hypot(uu, vv)**2/(2*D0**2)))
# img_f = np.fft.fft2(img)
# img_filtered_s_3 = np.fft.ifft2(img_f*filter_f).real

# https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
# img_filtered_s_8bit = ((img_filtered_s/img_filtered_s.max())*255).astype(np.uint8)
# clahe1 = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cl1 = clahe1.apply(img_filtered_s_8bit)

# plt.figure(figsize=(12,7))
# plt.subplot(1,4,1)
# plt.imshow(img_filtered_s_0, cmap='gray')
# plt.subplot(1,4,2)
# plt.imshow(img_filtered_s_1, cmap='gray')
# plt.subplot(1,4,3)
# plt.imshow(img_filtered_s_2, cmap='gray')
# plt.subplot(1,4,4)
# plt.imshow(img_filtered_s_3, cmap='gray')
# plt.show()

img_filtered_s_2 = scale(img_filtered_s_2)
img_filtered_s_equalized = exposure.equalize_adapthist(img_filtered_s_2,clip_limit=0.00015,nbins=26200)
img_filtered_s_equalized_gamma = img_filtered_s_equalized**1.2

# plt.figure(figsize=(12,7))
# plt.subplot(1,2,1)
# plt.imshow(img, cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(img_filtered_s_equalized, cmap='gray')
# plt.show()

np.savetxt('./outputs/method3' + '.txt', img_filtered_s_equalized_gamma, delimiter='\t')
