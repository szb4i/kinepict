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
from skimage import exposure
from utils.kernels.kirsch_kernels import *

### make patterns
# v = np.fft.fftfreq(400)
# u = np.fft.fftfreq(400)
# vv, uu = np.meshgrid(v, u, indexing='ij')
# pattern = np.where((np.hypot(uu,vv) > 0.4) & (np.hypot(uu,vv) < 0.42), 1, 0)
# C0 = 0.225
# W = 0.01
# pattern = np.exp(-(np.hypot(uu, vv) - C0)**2/(W**2))
# np.savetxt('./outputs/smart_sobel/test_pattern_circle.txt', np.fft.ifftshift(pattern), delimiter='\t')
# plt.imshow(np.fft.ifftshift(circle), cmap='gray')


### read img
# img = read_dva('./data/23_kep_test/Prostate/US_9660096_24_1.IMA')
# img = read_dva('./data/23_kep_test/X-ray 70%/hasE.IMA')
img = read_dva('./data/23_kep_test/Prostate/US_9660096_24_1.IMA')
# img = plt.imread('./data/tree.jpg')[:,:,0]
# img = np.loadtxt('./outputs/smart_sobel/test_pattern_circle.txt')
# img = np.loadtxt('./outputs/smart_sobel/test_pattern_gauss.txt')
# img = np.loadtxt('./outputs/smart_sobel/test_pattern_circle_gauss.txt')

### clahe
img = scale(img)
img_clahe = exposure.equalize_adapthist(img,kernel_size=[50,50],clip_limit=0.00015,nbins=26200)


### method1: sand glass shape
# kernel_sobel_x = get_sobel_x_kernel()
# kernel_sobel_y = get_sobel_y_kernel()
# threshold_pixel_value = 0.3
# threshold_pixel_diff = 0.15
# img_x_edge_plus = np.zeros(img.shape)
# img_gradient = convolve2d(img, kernel_sobel_x, mode='same', boundary = 'symm', fillvalue=0)
# for i in range(1,img.shape[0]-1):
#     for j in range(1,img.shape[1]-1):
#         if img_gradient[i,j] > threshold_pixel_value and (
#             abs(img_gradient[i-1,j-1] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i-1,j] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i-1,j+1] - img_gradient[i,j]) < threshold_pixel_diff or 
#             abs(img_gradient[i+1,j-1] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i+1,j] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i+1,j+1] - img_gradient[i,j]) < threshold_pixel_diff
#         ):
#             # img_x_edge_plus[i,j] = 1
#             img_x_edge_plus[i,j] = img[i,j]*1.8
#         else:
#             # img_x_edge_plus[i,j] = 0
#             img_x_edge_plus[i,j] = img[i,j]
# img_x_edge_minus = np.zeros(img.shape)
# img_gradient = convolve2d(img, kernel_sobel_x*-1, mode='same', boundary = 'symm', fillvalue=0)
# for i in range(1,img.shape[0]-1):
#     for j in range(1,img.shape[1]-1):
#         if img_gradient[i,j] > threshold_pixel_value and (
#             abs(img_gradient[i-1,j-1] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i-1,j] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i-1,j+1] - img_gradient[i,j]) < threshold_pixel_diff or 
#             abs(img_gradient[i+1,j-1] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i+1,j] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i+1,j+1] - img_gradient[i,j]) < threshold_pixel_diff
#         ):
#             # img_x_edge_minus[i,j] = 1
#             img_x_edge_minus[i,j] = img[i,j]*1.8
#         else:
#             # img_x_edge_minus[i,j] = 0
#             img_x_edge_minus[i,j] = img[i,j]
# img_y_edge_plus = np.zeros(img.shape)
# img_gradient = convolve2d(img, kernel_sobel_y, mode='same', boundary = 'symm', fillvalue=0)
# for i in range(1,img.shape[0]-1):
#     for j in range(1,img.shape[1]-1):
#         if img_gradient[i,j] > threshold_pixel_value and (
#             abs(img_gradient[i-1,j-1] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i,j-1] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i+1,j-1] - img_gradient[i,j]) < threshold_pixel_diff or 
#             abs(img_gradient[i-1,j+1] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i,j+1] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i+1,j+1] - img_gradient[i,j]) < threshold_pixel_diff
#         ):
#             # img_y_edge_plus[i,j] = 1
#             img_y_edge_plus[i,j] = img[i,j]*1.8
#         else:
#             # img_y_edge_plus[i,j] = 0
#             img_y_edge_plus[i,j] = img[i,j]
# img_y_edge_minus = np.zeros(img.shape)
# img_gradient = convolve2d(img, kernel_sobel_y*-1, mode='same', boundary = 'symm', fillvalue=0)
# for i in range(1,img.shape[0]-1):
#     for j in range(1,img.shape[1]-1):
#         if img_gradient[i,j] > threshold_pixel_value and (
#             abs(img_gradient[i-1,j-1] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i,j-1] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i+1,j-1] - img_gradient[i,j]) < threshold_pixel_diff or 
#             abs(img_gradient[i-1,j+1] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i,j+1] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i+1,j+1] - img_gradient[i,j]) < threshold_pixel_diff
#         ):
#             # img_y_edge_minus[i,j] = 1
#             img_y_edge_minus[i,j] = img[i,j]*1.8
#         else:
#             # img_y_edge_minus[i,j] = 0
#             img_y_edge_minus[i,j] = img[i,j]
# img_sum = np.where(((1 == img_x_edge_plus) | (1 == img_x_edge_minus) | (1 == img_y_edge_plus) | (1 == img_y_edge_minus)), 1, 0)
# img_sum = img_x_edge_plus + img_x_edge_minus + img_y_edge_plus + img_y_edge_minus


### method2:
# gradient direction, magnitude
kernel_sobel_x = get_sobel_x_kernel()
kernel_sobel_y = get_sobel_y_kernel()
img_gradient_x = convolve2d(img_clahe, kernel_sobel_x, mode='same', boundary = 'symm', fillvalue=0)
img_gradient_y = convolve2d(img_clahe, kernel_sobel_y, mode='same', boundary = 'symm', fillvalue=0)
img_gradient_direction = np.arctan2(img_gradient_y, img_gradient_x)
img_gradient_magnitude = np.hypot(img_gradient_x, img_gradient_y)
# gradient of gradient direction
img_gradient_direction_gradient_x = convolve2d(img_gradient_direction, kernel_sobel_x, mode='same', boundary = 'symm', fillvalue=0)
img_gradient_direction_gradient_y = convolve2d(img_gradient_direction, kernel_sobel_y, mode='same', boundary = 'symm', fillvalue=0)
img_gradient_direction_gradient = np.hypot(img_gradient_direction_gradient_x, img_gradient_direction_gradient_y)
# inversion: where directional change is small -> veins; where directional change is big -> noise. inverting it to have strong signal where veins are
img_gradient_direction_gradient_inverted = np.log(1/(img_gradient_direction_gradient+0.0001))
img_gradient_direction_gradient_inverted = scale(img_gradient_direction_gradient_inverted)

print('done')

plt.figure(figsize=(12,7))
plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.gca().set_title('img')
plt.subplot(1,3,2)
plt.imshow(img_clahe, cmap='gray')
plt.gca().set_title('clahe')
plt.subplot(1,3,3)
plt.imshow((1+3*img_gradient_direction_gradient_inverted)*img_clahe, cmap='gray')
plt.gca().set_title('gradient direction enhanced')
plt.show()

# np.savetxt('./outputs/smart_sobel/img_sum.txt', img_sum, delimiter='\t')