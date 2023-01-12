import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy import ndimage, misc
from scipy.ndimage import gaussian_filter

from utils.kernels.kirsch_kernels import *
from utils.kernels.laplacian_kernel import get_laplacian_kernel
from utils.file.dva_reader import read_dva
from utils.img.scaler import scale
from utils.kernels.sobel_kernels import get_sobel_x_kernel, get_sobel_y_kernel

### read dva
# img = read_dva('./data/23_kep_test/Prostate/US_9660096_24_1.IMA')
img = read_dva('./data/23_kep_test/X-ray 70%/hasE.IMA')
img_height = img.shape[0]
img_width = img.shape[1]

### wl operator
C = 9*np.identity(9)+np.full((9,9), -1)
normalizing_const = 1/(9**2)
kernel_dx = 1
kernel_dy = 1
img_wl = np.zeros((img_height, img_width))
for x in range(kernel_dy, img_height - kernel_dy):
  for y in range(kernel_dx, img_width - kernel_dx):
    region_flatten = img[x - kernel_dx: x + kernel_dx + 1, y - kernel_dy: y + kernel_dy + 1].flatten()
    img_wl[x,y] = normalizing_const*(region_flatten.dot(C.dot(region_flatten)))

### sobel
kernel_sobel_x = get_sobel_x_kernel()
kernel_sobel_y = get_sobel_y_kernel()
gx = convolve2d(img, kernel_sobel_x, mode='same', boundary = 'symm', fillvalue=0)
gy = convolve2d(img, kernel_sobel_y, mode='same', boundary = 'symm', fillvalue=0)
img_sobel = np.hypot(gx, gy)

# np.savetxt('./outputs/adaptive/wl.txt', img_wl, delimiter='\t')
# np.savetxt('./outputs/adaptive/sobel.txt', img_sobel, delimiter='\t')

plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
plt.imshow(img_sobel ** 0.6, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(img_wl ** 0.3 , cmap='gray')
plt.show()