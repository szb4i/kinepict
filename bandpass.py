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
from utils.kernels.laplacian_kernel import get_laplacian_kernel
from utils.kernels.gaussian_kernel import get_gaussian_kernel
from utils.file.dva_reader import read_dva
from utils.img.scaler import scale

### read dva
# img = read_dva('./data/23_kep_test/Carotis 100%/CAR17IM0')
img = read_dva('./data/23_kep_test/Prostate/US_9660096_24_1.IMA')
img_f = np.fft.fft2(img)

### bandpass gauss
v = np.fft.fftfreq(img.shape[0])
u = np.fft.fftfreq(img.shape[1])
vv, uu = np.meshgrid(v, u, indexing='ij')
C0 = 0.25
W = 0.1
filter_f = np.exp(-(np.hypot(uu, vv) - C0)**2/(W**2))
img_filtered_1 = np.fft.ifft2(img_f * filter_f).real

### difference of gausssains with numpy
sigma_1 = 0.19
gauss_1 = np.exp(-(np.hypot(uu, vv))**2/(2*sigma_1**2))
sigma_2 = 0.2
gauss_2 = np.exp(-(np.hypot(uu, vv))**2/(2*sigma_2**2))
gauss_diff = gauss_2 - gauss_1
img_filtered_2 = np.fft.ifft2(img_f * gauss_diff).real

### difference of gausssains with skimage
img_filtered_3 = difference_of_gaussians(img, 0.19, 0.2)

### plotting filters
plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
plt.imshow(np.fft.ifftshift(filter_f), cmap='gray')
plt.gca().set_title('bandpass')
plt.subplot(1,2,2)
plt.imshow(np.fft.ifftshift(gauss_diff), cmap='gray')
plt.gca().set_title('gauss diff')
plt.show()

### comparing filtered images
plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
plt.imshow(img_filtered_1, cmap='gray')
plt.gca().set_title('bandpass')
plt.subplot(1,2,2)
plt.imshow(img_filtered_2, cmap='gray')
plt.gca().set_title('gauss diff')
plt.show()

### high-pass emphasis
k_factor = 8
filter_empahis_f = 1 + k_factor*(gauss_diff)
img_filtered_4 = np.fft.ifft2(img_f*filter_empahis_f).real
plt.figure(figsize=(12,7))
plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.gca().set_title('original')
plt.subplot(1,3,2)
plt.imshow(img_filtered_2, cmap='gray')
plt.gca().set_title('bandpass')
plt.subplot(1,3,3)
plt.imshow(img_filtered_4, cmap='gray')
plt.gca().set_title('high-freq empahis')
plt.show()

# np.savetxt('./outputs/bandpass/img.txt', img, delimiter='\t')
# np.savetxt('./outputs/bandpass/img_filtered_4.txt', img_filtered_4, delimiter='\t')
