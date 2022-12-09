import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import cv2 as cv
from scipy import ndimage, misc
from scipy.ndimage import gaussian_filter

from utils.kernels.kirsch_kernels import *
from utils.kernels.laplacian_kernel import get_laplacian_kernel
from utils.kernels.box_kernel import get_box_kernel
from utils.file.dva_reader import read_dva
from utils.img.scaler import scale

### read dva
img = read_dva('./data/PATIENT_28_1.XA.0001.0001.2020.05.26.07.40.37.199459.139512372.IMA')
img_height = img.shape[0]
img_width = img.shape[1]
# plt.imshow(img, cmap='gray')
# plt.show()

### gaussian blur
img_blured = gaussian_filter(img, sigma=3)
# kernel_box = get_box_kernel(3)
# img_blured = convolve2d(img, kernel_box, mode='same', boundary = 'symm', fillvalue=0)
### edge enhancement
kernel_dx = 1
kernel_dy = 1
img_edge = np.zeros((img_height, img_width))
img_edge_1 = np.zeros((img_height, img_width))
for x in range(kernel_dy, img_height - kernel_dy):
  for y in range(kernel_dx, img_width - kernel_dx):
    partition = img_blured[x - kernel_dx: x + kernel_dx + 1, y - kernel_dy: y + kernel_dy + 1]
    img_edge[x,y] = partition.max() - partition.min()
# plt.figure(figsize=(12,7))
# plt.subplot(1,2,1)
# plt.imshow(img, cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(img_edge, cmap='gray')
# plt.show()
np.savetxt('./outputs/method5' + '.txt', img_edge, delimiter='\t')
