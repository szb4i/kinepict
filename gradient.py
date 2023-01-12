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

### read dva
# img = read_dva('./data/23_kep_test/Prostate/US_9660096_24_1.IMA')
img = read_dva('./data/23_kep_test/X-ray 70%/hasE.IMA')

### sobel filter
# https://fengl.org/2014/08/27/a-simple-implementation-of-sobel-filtering-in-python/
kernel_sobel_x = get_sobel_x_kernel()
kernel_sobel_y = get_sobel_y_kernel()
gx = convolve2d(img, kernel_sobel_x, mode='same', boundary = 'symm', fillvalue=0)
gy = convolve2d(img, kernel_sobel_y, mode='same', boundary = 'symm', fillvalue=0)
img_sobel = np.hypot(gx, gy)

### second order derivative: laplacian
kernel_laplacian = get_laplacian_kernel()
img_laplacian = convolve2d(img, kernel_laplacian, mode='same', boundary = 'symm', fillvalue=0)
img_laplacian = scale(img_laplacian, np.amax(img))

### combinging 1st and 2nd gradinet
kernel_box = get_box_kernel(5)
img_sobel_boxed = convolve2d(img_sobel, kernel_box, mode='same', boundary = 'symm', fillvalue=0)
img_final = img + ((img + img_laplacian) * img_sobel_boxed)
img_final = img_final**0.5

# img_clahe = exposure.equalize_adapthist(img + 2*img_final,kernel_size=[50,50],clip_limit=0.00015,nbins=26200)

plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(img_final, cmap='gray')
plt.show()

# np.savetxt('./outputs/gradient/img.txt', img, delimiter='\t')
# np.savetxt('./outputs/gradient/img_filtered.txt', img_final, delimiter='\t')
