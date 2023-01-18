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

### read img
# img = read_dva('./data/23_kep_test/Prostate/US_9660096_24_1.IMA')
# img = read_dva('./data/23_kep_test/X-ray 70%/hasE.IMA')
img = read_dva('./data/23_kep_test/Prostate/US_9660096_24_1.IMA')
# img = plt.imread('./data/tree.jpg')[:,:,0]

### method1: sand glass shape
kernel_sobel_x = get_sobel_x_kernel()
kernel_sobel_y = get_sobel_y_kernel()
threshold_pixel_value = 100
threshold_pixel_diff = 50
img_x_edge_plus = np.zeros(img.shape)
img_gradient = convolve2d(img, kernel_sobel_x, mode='same', boundary = 'symm', fillvalue=0)
for i in range(1,img.shape[0]-1):
    for j in range(1,img.shape[1]-1):
        if img_gradient[i,j] > threshold_pixel_value and (
            abs(img_gradient[i-1,j-1] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i-1,j] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i-1,j+1] - img_gradient[i,j]) < threshold_pixel_diff or 
            abs(img_gradient[i+1,j-1] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i+1,j] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i+1,j+1] - img_gradient[i,j]) < threshold_pixel_diff
        ):
            img_x_edge_plus[i,j] = 1
        else:
            img_x_edge_plus[i,j] = 0
img_x_edge_minus = np.zeros(img.shape)
img_gradient = convolve2d(img, kernel_sobel_x*-1, mode='same', boundary = 'symm', fillvalue=0)
for i in range(1,img.shape[0]-1):
    for j in range(1,img.shape[1]-1):
        if img_gradient[i,j] > threshold_pixel_value and (
            abs(img_gradient[i-1,j-1] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i-1,j] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i-1,j+1] - img_gradient[i,j]) < threshold_pixel_diff or 
            abs(img_gradient[i+1,j-1] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i+1,j] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i+1,j+1] - img_gradient[i,j]) < threshold_pixel_diff
        ):
            img_x_edge_minus[i,j] = 1
        else:
            img_x_edge_minus[i,j] = 0
img_y_edge_plus = np.zeros(img.shape)
img_gradient = convolve2d(img, kernel_sobel_y, mode='same', boundary = 'symm', fillvalue=0)
for i in range(1,img.shape[0]-1):
    for j in range(1,img.shape[1]-1):
        if img_gradient[i,j] > threshold_pixel_value and (
            abs(img_gradient[i-1,j-1] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i,j-1] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i+1,j-1] - img_gradient[i,j]) < threshold_pixel_diff or 
            abs(img_gradient[i-1,j+1] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i,j+1] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i+1,j+1] - img_gradient[i,j]) < threshold_pixel_diff
        ):
            img_y_edge_plus[i,j] = 1
        else:
            img_y_edge_plus[i,j] = 0
img_y_edge_minus = np.zeros(img.shape)
img_gradient = convolve2d(img, kernel_sobel_y*-1, mode='same', boundary = 'symm', fillvalue=0)
for i in range(1,img.shape[0]-1):
    for j in range(1,img.shape[1]-1):
        if img_gradient[i,j] > threshold_pixel_value and (
            abs(img_gradient[i-1,j-1] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i,j-1] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i+1,j-1] - img_gradient[i,j]) < threshold_pixel_diff or 
            abs(img_gradient[i-1,j+1] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i,j+1] - img_gradient[i,j]) < threshold_pixel_diff or abs(img_gradient[i+1,j+1] - img_gradient[i,j]) < threshold_pixel_diff
        ):
            img_y_edge_minus[i,j] = 1
        else:
            img_y_edge_minus[i,j] = 0
img_sum = np.where(((1 == img_x_edge_plus) | (1 == img_x_edge_minus) | (1 == img_y_edge_plus) | (1 == img_y_edge_minus)), 1, 0)


### method2:
# kernel_sobel_x = get_sobel_x_kernel()
# kernel_sobel_y = get_sobel_y_kernel()
# img_gradient_x = convolve2d(img, kernel_sobel_x, mode='same', boundary = 'symm', fillvalue=0)
# img_gradient_y = convolve2d(img, kernel_sobel_y, mode='same', boundary = 'symm', fillvalue=0)
# img_gradient_direction = np.arctan(img_gradient_y/(img_gradient_x+0.0001))


print('done')
# np.savetxt('./outputs/smart_sobel/img_sum.txt', img_sum, delimiter='\t')