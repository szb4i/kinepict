import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

from utils.kernels.kirsch_kernels import *
from utils.kernels.laplacian_kernel import get_laplacian_kernel
from utils.file.dva_reader import read_dva
from utils.img.scaler import scale
from utils.img.convolve2d_adaptive import convolve2d_adaptive

### read dva
img = read_dva('./data/PATIENT_28_1.XA.0001.0001.2020.05.26.07.40.37.199459.139512372.IMA')
img = img/img.max()
img_min, img_max = np.amin(img), np.amax(img)
# plt.imshow(img, cmap='gray')
# plt.show()

### adaptive_laplacian
# sad: Sum of Absolute Diferences
# H,W = img.shape
# sad = np.zeros((H, W), dtype=float)
# for i in range(1,H-1):
#   for j in range(1,W-1):
#     sad_local_sum = 0
#     sad_local_max = 0
#     for k in range(-1,2):
#      for l in range(-1,2):
#       if not(k == 0 and l == 0):
#         sad_local_sum += img[i+k,j+l]
#         sad_local_max = img[i+k,j+l] if img[i+k,j+l] > sad_local_max else sad_local_max
#     sad[i,j] = sad_local_sum/sad_local_max
kernel = get_laplacian_kernel()
g = convolve2d_adaptive(img, kernel)
# transient improvement method can be added also
# fig, ax = plt.subplots(1,3, figsize=(12,7))
# ax[0].imshow(img, cmap='gray')
# ax[0].set_title('img')
# ax[1].imshow(g, cmap='gray')
# ax[1].set_title('g')
# ax[2].imshow(img + g, cmap='gray')
# ax[2].set_title('img + g')
# plt.show()
plt.imshow(img + g, cmap='gray')
# plt.show()
plt.savefig('./outputs/laplacian_adaptive.png')
      