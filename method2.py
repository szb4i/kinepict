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
img_min, img_max = np.amin(img), np.amax(img)

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
# # fig, ax = plt.subplots(2,4, figsize=(12,7))
# # ax[0, 0].imshow(img, cmap='gray')
# # ax[0, 0].set_title('img')
# # ax[0, 1].imshow(laplacian_img, cmap='gray', vmin = img_min, vmax = img_max)
# # ax[0, 1].set_title('laplacian_img')
# # ax[0, 2].imshow(sum_laplacian_pure_img, cmap='gray', vmin = img_min, vmax = img_max)
# # ax[0, 2].set_title('sum_laplacian_pure_img')
# # ax[0, 3].imshow(sobel_img, cmap='gray', vmin = img_min, vmax = img_max)
# # ax[0, 3].set_title('sobel_img')
# # ax[1, 0].imshow(boxed_sobel_img, cmap='gray', vmin = img_min, vmax = img_max)
# # ax[1, 0].set_title('boxed_sobel_img')
# # ax[1, 1].imshow(product, cmap='gray')
# # ax[1, 1].set_title('product')
# # ax[1, 2].imshow(final_img, cmap='gray')
# # ax[1, 2].set_title('final_img')
# # ax[1, 3].imshow(gamma_corrected_final_img, cmap='gray')
# # ax[1, 3].set_title('gamma_corrected_final_img')
# # fig, ax = plt.subplots(1,2, figsize=(12,5))
# # ax[0].imshow(img, cmap='gray')
# # ax[1].imshow(gamma_corrected_final_img, cmap='gray')
# # plt.show()
from PIL import Image as im
gamma_corrected_final_img = ((gamma_corrected_final_img/gamma_corrected_final_img.max())*255).astype(np.uint8)
data = im.fromarray(gamma_corrected_final_img)
data.save('method2.png')