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
img = ((img/img.max())*255).astype(np.uint8)
img_height = img.shape[0]
img_width = img.shape[1]
# plt.imshow(img, cmap='gray')
# plt.show()

### gaussian
img_blured = gaussian_filter(img, sigma=3)
### wl operator
C = 9*np.identity(9)+np.full((9,9), -1)
kernel_dx = 1
kernel_dy = 1
output_wl = np.zeros((img_height, img_width))
for x in range(kernel_dy, img_height - kernel_dy):
  for y in range(kernel_dx, img_width - kernel_dx):
    kernel_local = C.dot(img_blured[x - kernel_dx: x + kernel_dx + 1, y - kernel_dy: y + kernel_dy + 1].flatten())
    output_wl[x, y] = 0.1111111111111111*(kernel_local.dot(img_blured[x - kernel_dx: x + kernel_dx + 1, y - kernel_dy: y + kernel_dy + 1].flatten()))
# mean_global = 1/(img_height*img_width)*output.sum()
# output_thresholded = (output > mean_global).astype(int)
### canny
output_8bit = ((output_wl/output_wl.max())*255).astype(np.uint8)
output_canny = cv.Canny(output_8bit,10,30)
### laplacian on output canny max
kernel_laplacian = get_laplacian_kernel()
output_canny_bool = output_canny.astype(bool)
output_final = np.zeros((img_height, img_width))
for x in range(kernel_dy, img_height - kernel_dy):
  for y in range(kernel_dx, img_width - kernel_dx):
    if output_canny_bool[x,y]:
      output_final[x, y] = (kernel_laplacian*img[x - kernel_dx: x + kernel_dx + 1, y - kernel_dy: y + kernel_dy + 1]).sum()
    else:
      output_final[x,y] = img[x,y]

# fig, ax = plt.subplots(1,3, figsize=(12,7))
# ax[0].imshow(img, cmap='gray')
# ax[0].set_title('img')
# ax[1].imshow(output_canny, cmap='gray')
# ax[1].set_title('output_canny')
# ax[2].imshow(output_final, cmap='gray')
# ax[2].set_title('output_final')
# plt.show()
plt.imshow(output_final, cmap='gray')
plt.savefig('./outputs/method1.png')
# plt.show()