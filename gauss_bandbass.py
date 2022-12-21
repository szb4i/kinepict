import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy import ndimage, misc
from skimage import exposure
import os
from skimage.filters import difference_of_gaussians, window

from utils.patterns.rectangle import get_rectangle
from utils.kernels.sobel_kernels import get_sobel_x_kernel, get_sobel_y_kernel
from utils.kernels.zeropad_kernel import get_zeropadded_kernel
from utils.kernels.gaussian_kernel import get_gaussian_kernel
from utils.file.dva_reader import read_dva
from utils.img.scaler import scale

### read dva
img = read_dva('./data/23_kep_test/Carotis 100%/CAR17IM0')
# np.savetxt('./outputs/img' + '.txt', img, delimiter='\t')
# plt.imshow(img, cmap='gray')
# plt.show()

### bandpass gauss
v = np.fft.fftfreq(img.shape[0])
u = np.fft.fftfreq(img.shape[1])
vv, uu = np.meshgrid(v, u, indexing='ij')
# C0 = 0.2
# W = 0.1
# filter_f = np.exp(-(np.hypot(uu, vv) - C0)**2/(W**2))
# img_filtered = np.fft.ifft2(np.fft.fft2(img) * filter_f).real
C0 = 0.25
W = 0.1
filter_f = np.exp(-(np.hypot(uu, vv) - C0)**2/(W**2))
img_filtered = np.fft.ifft2(np.fft.fft2(img) * filter_f).real

### difference of gausssains
## skimage
img_filtered = difference_of_gaussians(img, 0.19, 0.2)
## numpy
sigma_1 = 0.19
gauss_1 = np.exp(-(np.hypot(uu, vv))**2/(2*sigma_1**2))
sigma_2 = 0.2
gauss_2 = np.exp(-(np.hypot(uu, vv))**2/(2*sigma_2**2))
gauss_diff = gauss_2 - gauss_1
img_filtered = np.fft.ifft2(np.fft.fft2(img) * gauss_diff).real

# plt.imshow(np.fft.fftshift(filter_f), cmap='gray')
plt.figure(figsize=(7,7))
# plt.title('C0: ' + str(C0) + '; W: ' + str(W))
plt.imshow(img_filtered, cmap='gray')
plt.show()
# np.savetxt('./outputs/gauss/gauss' + '.txt', img_filtered, delimiter='\t')

