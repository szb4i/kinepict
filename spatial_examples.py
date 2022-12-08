import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

from utils.kernels.gaussian_kernel import get_gaussian_kernel
from utils.kernels.sobel_kernels import get_sobel_x_kernel, get_sobel_y_kernel
from utils.kernels.laplacian_kernel import get_laplacian_kernel
from utils.kernels.box_kernel import get_box_kernel
from utils.patterns.zone_plate_pattern import get_zone_plate_pattern
from utils.file.dicom_reader import read_dicom
from utils.file.dva_reader import read_dva
from utils.img.scaler import scale

### reading image
# img = plt.imread('./data/se011.png')
# img_min, img_max = np.amin(img), np.amax(img)
# plt.imshow(img, cmap='gray')
# plt.show()

### read dicom
# img = read_dicom('./data/PATIENT_1_AORTOILIAC_DVA2.dcm')
# img_min, img_max = np.amin(img), np.amax(img)
# plt.imshow(img, cmap='gray')
# plt.show()

### read dva
img = read_dva('./data/PATIENT_28_1.XA.0001.0001.2020.05.26.07.40.37.199459.139512372.IMA')
img_min, img_max = np.amin(img), np.amax(img)
# plt.imshow(img, cmap='gray')
# plt.show()
# from PIL import Image as im
# data = im.fromarray(img)
# data.save('gfg_dummy_pic.png')

### read dva unwrapped
# ima = read_dicom('./data/PATIENT_28_1.XA.0001.0001.2020.05.26.07.40.37.199459.139512372.IMA')
# dva_std_img = np.std(ima, axis=2, ddof=1)
# dva_mean_img = np.mean(ima, axis=2)
# fig, ax = plt.subplots(1,2, figsize=(12,5))
# ax[0].imshow(dva_std_img, cmap='gray')
# ax[1].imshow(dva_mean_img, cmap='gray')
# plt.show()


### histogram
# hist = np.histogram(img, bins=50)
# heights = hist[0]
# bins = hist[1]
# central_bins = (bins[1:]+bins[:-1])/2
# bin_width = central_bins[1] - central_bins[0]
# fig = plt.figure(figsize = (10, 5))
# plt.bar(central_bins, heights, bin_width)
# plt.show()

### highlighting colors between specific range
# img = np.where((img > 0.0) & (img < 0.4), 0, img)
# plt.figure(1)
# plt.imshow(img, cmap='gray')
# plt.show()

### conovolution
# https://medium.com/swlh/image-processing-with-python-convolutional-filters-and-kernels-b9884d91a8fd
# # kernel: box
# kernel = np.ones((3,3))
# # kernel: edge detecion
# kernel = np.array([[0, -1, 0],
#                     [-1, 4, -1],
#                     [0, -1, 0]])
# conv_img = convolve2d(img, kernel, mode='same')
# sharp_img = img + conv_img
# fig, ax = plt.subplots(2,2, figsize=(12,5))
# ax[0, 0].imshow(img, cmap='gray')
# ax[0, 1].imshow(conv_img, cmap='gray')
# ax[1, 0].imshow(sharp_img, cmap='gray', vmin = img_min, vmax = img_max)
# plt.show()


### masking
# kernel = get_gaussian_kernel()
# conv_img = convolve2d(img, kernel, mode='same')
# mask = img - conv_img
# k_weight = 10
# sharp = img + k_weight*mask
# fig, ax = plt.subplots(2,2, figsize=(12,5))
# ax[0, 0].imshow(img, cmap='gray')
# ax[0, 1].imshow(conv_img, cmap='gray')
# ax[1, 0].imshow(mask, cmap='gray')
# ax[1, 1].imshow(sharp, cmap='gray', vmin = img_min, vmax = img_max)
# plt.show()

### first order derivative: sobel
# https://fengl.org/2014/08/27/a-simple-implementation-of-sobel-filtering-in-python/
# kernel_x = get_sobel_x_kernel()
# kernel_y = get_sobel_y_kernel()
# gx = convolve2d(img, kernel_x, mode='same', boundary = 'symm', fillvalue=0)
# gy = convolve2d(img, kernel_y, mode='same', boundary = 'symm', fillvalue=0)
# magnitude = np.hypot(gx, gy)
# # fig, ax = plt.subplots(2,2, figsize=(12,5))
# # ax[0, 0].imshow(img, cmap='gray')
# # ax[0, 1].imshow(gx, cmap='gray')
# # ax[1, 0].imshow(gy, cmap='gray')
# # ax[1, 1].imshow(magnitude, cmap='gray')
# # plt.show()
# plt.imshow(magnitude, cmap='gray')
# plt.show()
# # plt.savefig('./outputs/sobel.png')

### second order derivative: laplacian
# kernel = get_laplacian_kernel()
# g = convolve2d(img, kernel, mode='same', boundary = 'symm', fillvalue=0)
# plt.imshow(img+g, cmap='gray')
# plt.show()
# plt.savefig('./outputs/laplacian.png')

### zoneplate
# zone_plate = get_zone_plate_pattern()
# plt.figure()
# plt.imshow(zone_plate, cmap='gray')
# plt.show()

### combining spatial enhancement methods
kernel_laplacian = get_laplacian_kernel()
laplacian_img = convolve2d(img, kernel_laplacian, mode='same')
laplacian_img = scale(laplacian_img, img_max)
sum_laplacian_pure_img = img + laplacian_img
kernel_sobel_x = get_sobel_x_kernel()
kernel_sobel_y = get_sobel_y_kernel()
gx = convolve2d(img, kernel_sobel_x, mode='same', boundary = 'symm', fillvalue=0)
gy = convolve2d(img, kernel_sobel_y, mode='same', boundary = 'symm', fillvalue=0)
sobel_img = np.hypot(gx, gy)
kernel_box = get_box_kernel(5)
boxed_sobel_img = convolve2d(sobel_img, kernel_box, mode='same', boundary = 'symm', fillvalue=0)
# boxed_sobel_img = gaussian_filter(sobel_img, sigma=1)
product = sum_laplacian_pure_img * boxed_sobel_img
final_img = img + product
gamma_corrected_final_img = np.array(final_img ** 0.5)
# fig, ax = plt.subplots(2,4, figsize=(12,7))
# ax[0, 0].imshow(img, cmap='gray')
# ax[0, 0].set_title('img')
# ax[0, 1].imshow(laplacian_img, cmap='gray', vmin = img_min, vmax = img_max)
# ax[0, 1].set_title('laplacian_img')
# ax[0, 2].imshow(sum_laplacian_pure_img, cmap='gray', vmin = img_min, vmax = img_max)
# ax[0, 2].set_title('sum_laplacian_pure_img')
# ax[0, 3].imshow(sobel_img, cmap='gray', vmin = img_min, vmax = img_max)
# ax[0, 3].set_title('sobel_img')
# ax[1, 0].imshow(boxed_sobel_img, cmap='gray', vmin = img_min, vmax = img_max)
# ax[1, 0].set_title('boxed_sobel_img')
# ax[1, 1].imshow(product, cmap='gray')
# ax[1, 1].set_title('product')
# ax[1, 2].imshow(final_img, cmap='gray')
# ax[1, 2].set_title('final_img')
# ax[1, 3].imshow(gamma_corrected_final_img, cmap='gray')
# ax[1, 3].set_title('gamma_corrected_final_img')
# fig, ax = plt.subplots(1,2, figsize=(12,5))
# ax[0].imshow(img, cmap='gray')
# ax[1].imshow(gamma_corrected_final_img, cmap='gray')
# plt.show()
plt.imshow(gamma_corrected_final_img, cmap='gray')
# plt.show()
plt.savefig('./outputs/method2.png')

