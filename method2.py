import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from PIL import Image as im

from utils.kernels.gaussian_kernel import get_gaussian_kernel
from utils.kernels.sobel_kernels import get_sobel_x_kernel, get_sobel_y_kernel
from utils.kernels.laplacian_kernel import get_laplacian_kernel
from utils.kernels.box_kernel import get_box_kernel
from utils.patterns.zone_plate_pattern import get_zone_plate_pattern
from utils.file.dicom_reader import read_dicom
from utils.file.dva_reader import read_dva
from utils.img.scaler import scale

### read dva
img = read_dva('./data/PATIENT_28_1.XA.0001.0001.2020.05.26.07.40.37.199459.139512372.IMA')

### combining spatial enhancement methods
kernel_laplacian = get_laplacian_kernel()
laplacian_img = convolve2d(img, kernel_laplacian, mode='same')
laplacian_img[laplacian_img < 0] = 0
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
img_product_sum = img + product
img_product_sum_gamma = np.array(img_product_sum ** 0.5)
# plt.figure(figsize=(12,7))
# plt.subplot(1,2,1)
# plt.imshow(img, cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(img_product_sum_gamma, cmap='gray')
# plt.show()
np.savetxt('./outputs/method2' + '.txt', img_product_sum_gamma, delimiter='\t')
