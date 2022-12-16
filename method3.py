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
img = read_dva('./data/23_kep_test/Vena I/1_1_N')
# np.savetxt('./outputs/img' + '.txt', img, delimiter='\t')
# plt.imshow(img, cmap='gray')
# plt.show()

### high frequency emphasis
v = np.fft.fftfreq(img.shape[0])
u = np.fft.fftfreq(img.shape[1])
vv, uu = np.meshgrid(v, u, indexing='ij')
k1 = 0.5  
k2 = 0.75
D0 = 0.05
filter_f_1 = k1 + k2*(1 - np.exp(-np.hypot(uu, vv)**2/(2*D0**2)))
img_f_1 = np.fft.fft2(img)
img_filtered_s_1 = np.fft.ifft2(img_f_1*filter_f_1).real
img_filtered_s_1 = scale(img_filtered_s_1)
img_filtered_s_1_equalized = exposure.equalize_adapthist(img_filtered_s_1,clip_limit=0.00015,nbins=26200)
img_filtered_s_1_equalized_gamma = img_filtered_s_1_equalized**1.2

filter_f_2 = np.where(np.hypot(uu,vv) < 0.4, k1 + k2*(1 - np.exp(-np.hypot(uu, vv)**2/(2*D0**2))), 0)
img_f_2 = np.fft.fft2(img)
img_filtered_s_2 = np.fft.ifft2(img_f_2*filter_f_2).real
img_filtered_s_2 = scale(img_filtered_s_2)
img_filtered_s_2_equalized = exposure.equalize_adapthist(img_filtered_s_2,clip_limit=0.00015,nbins=26200)
img_filtered_s_2_equalized_gamma = img_filtered_s_2_equalized**1.2

# plt.imshow(np.fft.fftshift(filter_f_2), cmap='gray')

# https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
# img_filtered_s_8bit = ((img_filtered_s/img_filtered_s.max())*255).astype(np.uint8)
# clahe1 = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cl1 = clahe1.apply(img_filtered_s_8bit)

plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
plt.imshow(img_filtered_s_1_equalized, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(img_filtered_s_2_equalized, cmap='gray')
plt.show()

# np.savetxt('./outputs/method3' + '.txt', img_filtered_s_equalized_gamma, delimiter='\t')
