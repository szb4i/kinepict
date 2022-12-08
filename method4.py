import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import cv2 as cv
from scipy import ndimage, misc
from scipy.ndimage import gaussian_filter

from utils.kernels.kirsch_kernels import *
from utils.kernels.laplacian_kernel import get_laplacian_kernel
from utils.file.dva_reader import read_dva
from utils.img.scaler import scale

### read dva
img = read_dva('./data/PATIENT_28_1.XA.0001.0001.2020.05.26.07.40.37.199459.139512372.IMA')
img_height = img.shape[0]
img_width = img.shape[1]

### wl operator
C = 9*np.identity(9)+np.full((9,9), -1)
kernel_dx = 1
kernel_dy = 1
output_wl = np.zeros((img_height, img_width))
output_wl_1 = np.zeros((img_height, img_width))
for x in range(kernel_dy, img_height - kernel_dy):
  for y in range(kernel_dx, img_width - kernel_dx):
    region_flatten = img[x - kernel_dx: x + kernel_dx + 1, y - kernel_dy: y + kernel_dy + 1].flatten()
    output_wl[x,y] = 0.01234567*(region_flatten.dot(C.dot(region_flatten)))
np.savetxt('output_wl' + '.txt', output_wl, delimiter='\t')
