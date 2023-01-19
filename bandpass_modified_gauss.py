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
C0 = 0.225
W = 0.1
filter_f = np.exp(-(np.hypot(uu, vv) - C0)**2/(W**2))
img_filtered = np.fft.ifft2(img_f * filter_f).real

### gauss diff
sigma_1 = 0.19
gauss_1 = np.exp(-(np.hypot(uu, vv))**2/(2*sigma_1**2))
sigma_2 = 0.2
gauss_2 = np.exp(-(np.hypot(uu, vv))**2/(2*sigma_2**2))
gauss_diff = gauss_2 - gauss_1

### plotting filters
# plt.figure(figsize=(12,7))
# plt.subplot(2,1,1)
# plt.imshow(np.fft.ifftshift(filter_f), cmap='gray')
# plt.gca().set_title('bandpass')
# plt.subplot(2,1,2)
# plt.imshow(img_filtered, cmap='gray')
# plt.gca().set_title('gauss diff')
# plt.show()

### high-pass emphasis
k_factor = 80
filter_emphasis_f = 1 + k_factor*(filter_f)
img_filtered_emphasis = np.fft.ifft2(img_f*filter_emphasis_f).real
plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.gca().set_title('original')
plt.subplot(1,2,2)
plt.imshow(img_filtered_emphasis, cmap='gray')
plt.gca().set_title('bandpass')
plt.show()

# np.savetxt('./outputs/bandpass_modified_gauss/img_filtered.txt', img_filtered, delimiter='\t')
