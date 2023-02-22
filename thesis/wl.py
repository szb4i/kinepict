import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.signal import convolve2d
import sys
 
# setting path
sys.path.append('../')
from utils.kernels.sobel_kernels import get_sobel_x_kernel, get_sobel_y_kernel
from utils.img.scaler import scale

img = scale(plt.imread('../data/tree.jpg')[:,:,0])
noise = np.random.normal(0, 0.05  , img.shape)
img_noisy = img+noise

### wl operator
C = 9*np.identity(9)+np.full((9,9), -1)
normalizing_const = 1/(9**2)
kernel_dx = 1
kernel_dy = 1
img_wl = np.zeros((img.shape[0], img.shape[1]))
img_wl_noisy = np.zeros((img.shape[0], img.shape[1]))
for x in range(kernel_dy, img.shape[1] - kernel_dy):
  for y in range(kernel_dx, img.shape[0] - kernel_dx):
    region_flatten = img[x - kernel_dx: x + kernel_dx + 1, y - kernel_dy: y + kernel_dy + 1].flatten()
    img_wl[x,y] = normalizing_const*(region_flatten.dot(C.dot(region_flatten)))
    region_flatten = img_noisy[x - kernel_dx: x + kernel_dx + 1, y - kernel_dy: y + kernel_dy + 1].flatten()
    img_wl_noisy[x,y] = normalizing_const*(region_flatten.dot(C.dot(region_flatten)))
    
### sobel
kernel_x = get_sobel_x_kernel()
kernel_y = get_sobel_y_kernel()
gx = convolve2d(img, kernel_x, mode='same', boundary = 'symm', fillvalue=0)
gy = convolve2d(img, kernel_y, mode='same', boundary = 'symm', fillvalue=0)
img_sobel = np.hypot(gx, gy)
gx = convolve2d(img_noisy, kernel_x, mode='same', boundary = 'symm', fillvalue=0)
gy = convolve2d(img_noisy, kernel_y, mode='same', boundary = 'symm', fillvalue=0)
img_sobel_noisy = np.hypot(gx, gy)
    
# plt.subplot(1,2,1)
# plt.imshow(img_sobel, cmap='gray')
# plt.axis("off")
# plt.gca().set_title('sobel magnitude', fontdict={'fontsize':18})
# plt.subplot(1,2,2)
# plt.imshow(img_wl, cmap='gray')
# plt.axis("off")
# plt.gca().set_title('wl', fontdict={'fontsize':18})
# plt.tight_layout()
# plt.show()

plt.subplot(1,3,1)
plt.imshow(img_noisy, cmap='gray')
plt.axis("off")
plt.gca().set_title('img', fontdict={'fontsize':18})
plt.subplot(1,3,2)
plt.imshow(img_sobel_noisy, cmap='gray')
plt.axis("off")
plt.gca().set_title('sobel', fontdict={'fontsize':18})
plt.subplot(1,3,3)
plt.imshow(img_wl_noisy, cmap='gray')
plt.axis("off")
plt.gca().set_title('wl', fontdict={'fontsize':18})
plt.tight_layout()
plt.show()