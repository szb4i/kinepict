import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy import ndimage, misc

from utils.patterns.rectangle import get_rectangle
from utils.kernels.sobel_kernels import get_sobel_x_kernel, get_sobel_y_kernel
from utils.kernels.zeropad_kernel import get_zeropadded_kernel
from utils.kernels.gaussian_kernel import get_gaussian_kernel
from utils.file.dva_reader import read_dva
from utils.file.dva_reader import read_dva
from utils.img.scaler import scale
import cv2 as cv

### read image
# img = plt.imread('./data/tree.jpg')
# img = img[:,:,0]
# img = img/img.max()

### read dva
img = read_dva('./data/PATIENT_28_1.XA.0001.0001.2020.05.26.07.40.37.199459.139512372.IMA')
img = ((img/img.max())*255).astype(np.uint8)
# plt.imshow(img, cmap='gray')
# plt.show()

### rectangle
# rectangle = get_rectangle(512, 512, 10, 100)
# rectangle_ft = np.fft.fft2(rectangle)
# fig, ax = plt.subplots(1,2, figsize=(12,5))
# ax[0].imshow(rectangle, cmap='gray')
# ax[1].imshow(np.fft.fftshift(np.log(np.abs(rectangle_ft))), cmap='gray')
# plt.show()

### create filter in fourier space
# u = np.fft.fftfreq(img.shape[1])
# v = np.fft.fftfreq(img.shape[0])
# vv, uu = np.meshgrid(v, u, indexing='ij')
# filter = (np.hypot(vv, uu) < 0.01)
# img_ft = np.fft.fft2(img)
# img_filtered = np.fft.ifft2(img_ft*filter).real
# plt.imshow(img_filtered, vmin=-1.0, vmax=1.0, cmap='gray')
# plt.show()

### create kernel in real space. compare fourier filtered image with spatial filered image
# img_ft = np.fft.fft2(img)
# kernel = get_gaussian_kernel(l=15, sig=10000000)
# kernel_padded = get_zeropadded_kernel(img, kernel)
# kernel_ft = np.fft.fft2(kernel_padded)
# result_fourier = np.fft.ifft2(img_ft*kernel_ft).real
# spatial_transform = convolve2d(img, kernel, mode='same', boundary = 'symm', fillvalue=0)
# fig, ax = plt.subplots(1,2, figsize=(12,5))
# ax[0].imshow(result_fourier, cmap='gray')
# ax[1].imshow(spatial_transform, cmap='gray')
# plt.show()


### gaussian in fourier with scipy
# img_f = np.fft.fft2(img)
# img_filtered_f = ndimage.fourier_gaussian(img_f, sigma=5)
# img_filtered_s = np.fft.ifft2(img_filtered_f).real
# fig, ax = plt.subplots(1,2, figsize=(12,5))
# ax[0].imshow(img, cmap='gray')
# ax[1].imshow(img_filtered_s, cmap='gray')
# plt.show()

### gaussian in fourier with numpy
# v = np.fft.fftfreq(img.shape[0])
# u = np.fft.fftfreq(img.shape[1])
# vv, uu = np.meshgrid(v, u, indexing='ij')
# D0 = 1/10
# filter_f = np.exp(-np.hypot(uu, vv)**2/(2*D0**2))
# img_f = np.fft.fft2(img)
# img_filtered_s = np.fft.ifft2(img_f*filter_f).real
# fig, ax = plt.subplots(1,2, figsize=(12,5))
# ax[0].imshow(img, cmap='gray')
# ax[1].imshow(img_filtered_s, cmap='gray')
# plt.show()

### ideal low pass filter
# h, w = img.shape
# filter_f = np.zeros((h, w), dtype=np.float64)
# D0 = 10
# for u in range(h):
#     for v in range(w):
#         D = np.sqrt((u-h/2)**2 + (v-w/2)**2)
#         if D <= D0:
#             filter_f[u,v] = 1
#         else:
#             filter_f[u,v] = 0
# img_f = np.fft.fftshift(np.fft.fft2(img))
# img_filtered_f = img_f * filter_f
# img_filtered_s = np.fft.ifft2(np.fft.ifftshift(img_filtered_f))
# plt.imshow(np.abs(img_filtered_s), cmap='gray')
# plt.colorbar()
# plt.show()

### laplacian
# v = np.fft.fftfreq(img.shape[0])
# u = np.fft.fftfreq(img.shape[1])
# vv, uu = np.meshgrid(v, u, indexing='ij')
# filter_f = -4*np.pi**2*(uu**2+vv**2)
# img_f = np.fft.fft2(img)
# filter_s = np.fft.ifft2(img_f*filter_f).real
# filter_s = scale(filter_s, max=1, min=-1)
# c = -1
# img_filtered_s = img + c*filter_s
# img_filtered_s = np.clip(img_filtered_s, 0, 1)
# fig, ax = plt.subplots(1,2, figsize=(12,5))
# ax[0].imshow(img, cmap='gray')
# ax[1].imshow(img_filtered_s, cmap='gray')
# plt.show()

### high frequency emphasis
v = np.fft.fftfreq(img.shape[0])
u = np.fft.fftfreq(img.shape[1])
vv, uu = np.meshgrid(v, u, indexing='ij')
D0 = 1/10
k1 = 0.5
k2 = 0.75
filter_f = k1 + k2*(1 - np.exp(-np.hypot(uu, vv)**2/(2*D0**2)))
img_f = np.fft.fft2(img)
img_filtered_s = np.fft.ifft2(img_f*filter_f).real
img_filtered_s_8bit = ((img_filtered_s/img_filtered_s.max())*255).astype(np.uint8)
clahe1 = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe1.apply(img_filtered_s_8bit)
plt.figure(figsize=(12,7))
plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.subplot(1,3,2)
plt.imshow(img_filtered_s_8bit, cmap='gray')
plt.subplot(1,3,3)
plt.imshow(cl1, cmap='gray')
plt.show()