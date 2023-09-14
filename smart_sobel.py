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
from utils.fusion.wavelet_fusion import fuse
from skimage import restoration

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
img = scale(img)
# img = plt.imread('./data/tree.jpg')[:,:,0]
# img = np.loadtxt('./outputs/smart_sobel/test_pattern_circle.txt')
# img = np.loadtxt('./outputs/smart_sobel/test_pattern_gauss.txt')
# img = np.loadtxt('./outputs/smart_sobel/test_pattern_circle_gauss.txt')

### clahe
# img = exposure.equalize_adapthist(img,kernel_size=[50,50],clip_limit=0.00015,nbins=26200)


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
img_gradient_x = convolve2d(img, kernel_sobel_x, mode='same', boundary = 'symm', fillvalue=0)
img_gradient_y = convolve2d(img, kernel_sobel_y, mode='same', boundary = 'symm', fillvalue=0)
img_gradient_direction = np.arctan2(img_gradient_y, img_gradient_x)
img_gradient_magnitude = scale(np.hypot(img_gradient_x, img_gradient_y))
# gradient of gradient direction
img_gradient_direction_gradient_x = convolve2d(img_gradient_direction, kernel_sobel_x, mode='same', boundary = 'symm', fillvalue=0)
img_gradient_direction_gradient_y = convolve2d(img_gradient_direction, kernel_sobel_y, mode='same', boundary = 'symm', fillvalue=0)
img_gradient_direction_gradient_mangitude = np.hypot(img_gradient_direction_gradient_x, img_gradient_direction_gradient_y)
# inversion: where directional change is small -> veins; where directional change is big -> noise. inverting it to have strong signal where veins are
img_gradient_direction_gradient_mangitude_inverted = np.log(1/(img_gradient_direction_gradient_mangitude+0.0001))
img_gradient_direction_gradient_mangitude_inverted = scale(img_gradient_direction_gradient_mangitude_inverted)

### fuse
# img_fused = fuse(img_gradient_magnitude, img_gradient_direction_gradient_inverted, fusion_method='mean')
img_sum = 2*img + (2*img_gradient_magnitude + img_gradient_direction_gradient_mangitude_inverted)

### plot
plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(img_sum, cmap='gray')
plt.show()

# np.savetxt('./outputs/smart_sobel/img_sum.txt', img_sum, delimiter='\t')

# ### img_sum_1: simple sum
# img_sum_1 = scale(img + img_gradient_direction_gradient_inverted)
# ### img_sum_2: clahe on original image and add it to directional image
# img_sum_2 = scale(exposure.equalize_adapthist(img,kernel_size=[50,50],clip_limit=0.00015,nbins=26200) + img_gradient_direction_gradient_inverted)
# ### img_sum_3 sum images and apply clahe on sum
# img_sum_3 = exposure.equalize_adapthist(scale(img + img_gradient_direction_gradient_inverted),kernel_size=[100,100],clip_limit=0.00015,nbins=26200)

# # ### snr fusion: works but very poor result
# img_to_analyze = img_gradient_direction_gradient_inverted
# roi_signal_in_vein_1 = img_to_analyze[749:757, 438:446]
# roi_signal_out_vein_1 = img_to_analyze[759:767, 423:431]
# roi_signal_in_vein_2 = img_to_analyze[645:652, 695:702]
# roi_signal_out_vein_2 = img_to_analyze[644:651, 707:714]
# roi_signal_in_vein_3 = img_to_analyze[910:916, 289:295]
# roi_signal_out_vein_3 = img_to_analyze[904:910, 299:305]
# signal_1 = np.mean(roi_signal_in_vein_1) - np.mean(roi_signal_out_vein_1)
# signal_2 = np.mean(roi_signal_in_vein_2) - np.mean(roi_signal_out_vein_2)
# signal_3 = np.mean(roi_signal_in_vein_3) - np.mean(roi_signal_out_vein_3)
# noise_1 = np.std(roi_signal_out_vein_1)
# noise_2 = np.std(roi_signal_out_vein_2)
# noise_3 = np.std(roi_signal_out_vein_3)
# snr_1_directional = signal_1/noise_1
# snr_2_directional = signal_2/noise_2
# snr_3_directional = signal_3/noise_3
# snr_directional_avg = (snr_1_directional + snr_2_directional + snr_3_directional)/3
# ### snr magnitude
# img_to_analyze = img_gradient_magnitude
# roi_signal_in_vein_1 = img_to_analyze[749:757, 438:446]
# roi_signal_out_vein_1 = img_to_analyze[759:767, 423:431]
# roi_signal_in_vein_2 = img_to_analyze[645:652, 695:702]
# roi_signal_out_vein_2 = img_to_analyze[644:651, 707:714]
# roi_signal_in_vein_3 = img_to_analyze[910:916, 289:295]
# roi_signal_out_vein_3 = img_to_analyze[904:910, 299:305]
# signal_1 = np.mean(roi_signal_in_vein_1) - np.mean(roi_signal_out_vein_1)
# signal_2 = np.mean(roi_signal_in_vein_2) - np.mean(roi_signal_out_vein_2)
# signal_3 = np.mean(roi_signal_in_vein_3) - np.mean(roi_signal_out_vein_3)
# noise_1 = np.std(roi_signal_out_vein_1)
# noise_2 = np.std(roi_signal_out_vein_2)
# noise_3 = np.std(roi_signal_out_vein_3)
# snr_1_magnitude = signal_1/noise_1
# snr_2_magnitude = signal_2/noise_2
# snr_3_magnitude = signal_3/noise_3
# snr_magnitude_avg = (snr_1_magnitude + snr_2_magnitude + snr_3_magnitude)/3
# ### add images with weigths:
# img_sum_weighted = (snr_directional_avg/(snr_directional_avg+snr_magnitude_avg))*img_gradient_direction_gradient_inverted + (snr_magnitude_avg/(snr_directional_avg+snr_magnitude_avg))*img_gradient_magnitude