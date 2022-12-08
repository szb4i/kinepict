import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

from utils.kernels.kirsch_kernels import *
from utils.file.dva_reader import read_dva
from utils.img.scaler import scale

### read dva
img = read_dva('./data/PATIENT_28_1.XA.0001.0001.2020.05.26.07.40.37.199459.139512372.IMA')
img = img/img.max()
img_min, img_max = np.amin(img), np.amax(img)
# plt.imshow(img, cmap='gray')
# plt.show()

### kirsch kernels
kernel_0 = get_kirsch_kernel_0()
kernel_45 = get_kirsch_kernel_45()
kernel_90 = get_kirsch_kernel_90()
kernel_135 = get_kirsch_kernel_135()
kernel_180 = get_kirsch_kernel_180()
kernel_225 = get_kirsch_kernel_225()
kernel_270 = get_kirsch_kernel_270()
kernel_315 = get_kirsch_kernel_315()
g_0 = convolve2d(img, kernel_0, mode='same', boundary = 'symm', fillvalue=0)
g_45 = convolve2d(img, kernel_45, mode='same', boundary = 'symm', fillvalue=0)
g_90 = convolve2d(img, kernel_90, mode='same', boundary = 'symm', fillvalue=0)
g_90 = convolve2d(img, kernel_90, mode='same', boundary = 'symm', fillvalue=0)
g_135 = convolve2d(img, kernel_135, mode='same', boundary = 'symm', fillvalue=0)
g_180 = convolve2d(img, kernel_180, mode='same', boundary = 'symm', fillvalue=0)
g_225 = convolve2d(img, kernel_225, mode='same', boundary = 'symm', fillvalue=0)
g_270 = convolve2d(img, kernel_270, mode='same', boundary = 'symm', fillvalue=0)
g_315 = convolve2d(img, kernel_315, mode='same', boundary = 'symm', fillvalue=0)
g_max = np.stack([g_0, g_45, g_90, g_90, g_135, g_180, g_225, g_270, g_315], axis=2).max(axis=2)
### start: comparing neighborhood points of g_max
# H,W = img.shape
# mi_w = np.zeros((H, W), dtype=bool)
# mi_h = np.zeros((H, W), dtype=bool)
# for i in range(H):
#   for j in range(W):
#     if j>0 and j<W-1:
#       mi_w[i, j] = g_max[i, j-1] < g_max[i, j] and g_max[i, j] > g_max[i, j+1]
#     if i>0 and i<H-1:
#       mi_h[i, j] = g_max[i-1 ,j] < g_max[i, j] and g_max[i, j] > g_max[i+1, j]
# mi_product = mi_w*mi_h
# mi_max = mi_product*g_max
### end: comparing neighborhood points of g_max
# fig, ax = plt.subplots(2,5, figsize=(12,7))
# ax[0,0].imshow(img, cmap='gray', vmin = img_min, vmax = img_max)
# ax[0, 0].set_title('img')
# ax[0,1].imshow(g_0, cmap='gray', vmin = img_min, vmax = img_max)
# ax[0,1].set_title('g_0')
# ax[0,2].imshow(g_45, cmap='gray', vmin = img_min, vmax = img_max)
# ax[0,2].set_title('g_45')
# ax[0,3].imshow(g_90, cmap='gray', vmin = img_min, vmax = img_max)
# ax[0,3].set_title('g_90')
# ax[0,4].imshow(g_135, cmap='gray', vmin = img_min, vmax = img_max)
# ax[0,4].set_title('g_135')
# ax[1,0].imshow(g_180, cmap='gray', vmin = img_min, vmax = img_max)
# ax[1,0].set_title('g_180')
# ax[1,1].imshow(g_225, cmap='gray', vmin = img_min, vmax = img_max)
# ax[1,1].set_title('g_225')
# ax[1,2].imshow(g_270, cmap='gray', vmin = img_min, vmax = img_max)
# ax[1,2].set_title('g_270')
# ax[1,3].imshow(g_315, cmap='gray', vmin = img_min, vmax = img_max)
# ax[1,3].set_title('g_315')
# ax[1,4].imshow(g_max, cmap='gray')
# ax[1,4].set_title('g_max')
# plt.show()
# fig, ax = plt.subplots(1,3, figsize=(12,7))
# ax[0].imshow(img, cmap='gray')
# ax[0].set_title('img')
# ax[1].imshow(g_max, cmap='gray')
# ax[1].set_title('g_max')
# ax[2].imshow(img + g_max, cmap='gray')
# ax[2].set_title('img + g_max')
# plt.show()
# fig, ax = plt.subplots(1,2, figsize=(12,7))
# ax[0].imshow(img, cmap='gray', vmin = img_min, vmax = img_max)
# ax[0].set_title('img')
# ax[1].imshow(mi_max, cmap='gray')
# ax[1].set_title('mi_max')
# plt.show()
# plt.imshow(g_max, cmap='gray')
# plt.show()
from PIL import Image as im
data = im.fromarray(((g_max/g_max.max())*255).astype(np.uint8))
data.save('kisrch.png')


