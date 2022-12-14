import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import time

from utils.kernels.gaussian_kernel import get_gaussian_kernel
from utils.file.dva_reader import read_dva

start = time.time()

img = read_dva('./data/PATIENT_28_1.XA.0001.0001.2020.05.26.07.40.37.199459.139512372.IMA')

kernel = get_gaussian_kernel()

### method: numpy
img_sliding_window = np.lib.stride_tricks.sliding_window_view(img, (3,3))
# img_convolved = np.sum((img_sliding_window * kernel), axis=(2,3))
### method: scipy
# img_convolved = convolve2d(img, kernel, mode='same')
### method: for loop
# img_height = img.shape[0]
# img_width = img.shape[1]
# img_convolved = np.zeros((img_height, img_width))
# kernel_dx = 1
# kernel_dy = 1
# for x in range(kernel_dy, img_height - kernel_dy):
#   for y in range(kernel_dx, img_width - kernel_dx):
#     img_convolved[x,y] = np.sum(img[x - kernel_dx: x + kernel_dx + 1, y - kernel_dy: y + kernel_dy + 1]*kernel)

end = time.time()
print(end - start)

# plt.figure(figsize=(12,7))
# plt.subplot(1,2,1)
# plt.imshow(img, cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(img_convolved, cmap='gray')
# plt.show()
