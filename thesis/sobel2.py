import numpy as np
import sys
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

sys.path.append('../')
from utils.kernels.sobel_kernels import get_sobel_x_kernel, get_sobel_y_kernel
from utils.file.dva_reader import read_dva
from utils.img.scaler import scale

img = read_dva('../data/23_kep_test/Prostate/US_9660096_24_1.IMA')
img = scale(img)

kernel_sobel_x = get_sobel_x_kernel()
kernel_sobel_y = get_sobel_y_kernel()
gx = convolve2d(img, kernel_sobel_x, mode='same', boundary = 'symm', fillvalue=0)
gy = convolve2d(img, kernel_sobel_y, mode='same', boundary = 'symm', fillvalue=0)
sobel_img = np.hypot(gx, gy)

plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(sobel_img, cmap='gray')
plt.show()

