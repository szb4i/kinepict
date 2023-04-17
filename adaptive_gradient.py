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
img = read_dva('./data/23_kep_test/Prostate/US_9660096_24_1.IMA')
# img = read_dva('./data/23_kep_test/X-ray 70%/hasE.IMA')
img_height = img.shape[0]
img_width = img.shape[1]

### sobel
kernel_sobel_x = get_sobel_x_kernel()
kernel_sobel_y = get_sobel_y_kernel()
gx = convolve2d(img, kernel_sobel_x, mode='same', boundary = 'symm', fillvalue=0)
gy = convolve2d(img, kernel_sobel_y, mode='same', boundary = 'symm', fillvalue=0)
img_sobel = np.hypot(gx, gy)

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

### range
# img_blured = gaussian_filter(img, sigma=3)
img_sliding_window = np.lib.stride_tricks.sliding_window_view(img, (3,3))
img_sliding_window = np.reshape(img_sliding_window, (img_sliding_window.shape[0], img_sliding_window.shape[1], 9))
img_range = np.amax(img_sliding_window, axis=2) - np.amin(img_sliding_window, axis=2)

plt.figure(figsize=(12,7))
plt.suptitle("adaptive_gradient", size=14)
plt.subplot(1,3,1)
plt.imshow(img_sobel, cmap='gray')
plt.gca().set_title('sobel')
plt.subplot(1,3,2)
plt.imshow(img_wl , cmap='gray')
plt.gca().set_title('wl')
plt.subplot(1,3,3)
plt.imshow(img_range, cmap='gray')
plt.gca().set_title('statistical range')
plt.show()

# np.savetxt('./outputs/adaptive/wl.txt', img_wl, delimiter='\t')
# np.savetxt('./outputs/adaptive/sobel.txt', img_sobel, delimiter='\t')
# np.savetxt('./outputs/adaptive/range.txt', img_range, delimiter='\t')