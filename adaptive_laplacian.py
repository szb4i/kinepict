import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

from utils.kernels.kirsch_kernels import *
from utils.file.dva_reader import read_dva
from utils.img.scaler import scale
from utils.img.convolve2d_adaptive_laplacian import convolve2d_adaptive_laplacian

### read dva
# img = read_dva('./data/PATIENT_28_1.XA.0001.0001.2020.05.26.07.40.37.199459.139512372.IMA')
img = read_dva('./data/23_kep_test/X-ray 70%/hasE.IMA')
# img = -1*img
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
g = convolve2d_adaptive_laplacian(img)
# transient improvement method can be added also
plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(g, cmap='gray')
plt.show()
# plt.savefig('./outputs/laplacian_adaptive.png')
      