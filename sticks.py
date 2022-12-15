import numpy as np
import matplotlib.pyplot as plt
import math
import time

from utils.kernels.gaussian_kernel import get_gaussian_kernel
from utils.file.dva_reader import read_dva

start = time.time()

### read image
img = plt.imread('./data/tree.jpg')
img = img[:,:,0]
img = img/img.max()
noise = np.random.normal(0, 0.1, img.shape)
img = img+noise
img[img<0] = 0
img_height = img.shape[0]
img_width = img.shape[1]
# img = read_dva('./data/PATIENT_28_1.XA.0001.0001.2020.05.26.07.40.37.199459.139512372.IMA')
# img_height = img.shape[0]
# img_width = img.shape[1]

a = np.arange(1,7)
b = np.array([
  np.arange(11,20),
  np.arange(21,30),
  np.arange(31,40),
  np.arange(41,50),
  np.arange(51,60),
  np.arange(61,70),
  np.arange(71,80),
  np.arange(81,90),
  np.arange(91,100),
])
a_sliding_window = np.lib.stride_tricks.sliding_window_view(a, 3)
b_sliding_window = np.lib.stride_tricks.sliding_window_view(b, (3,3))
kernel = np.ones((3,3))

img_sliding_window = np.lib.stride_tricks.sliding_window_view(img, (7,7))

stick_0 = np.zeros((7,7))
stick_0[:4,3:] = np.array([
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [1, 1, 1, 1]
])
stick_1 = np.zeros((7,7))
stick_1[:4,3:] = np.array([
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 1, 1],
  [1, 1, 0, 0]
])
stick_2 = np.zeros((7,7))
stick_2[:4,3:] = np.array([
  [0, 0, 0, 0],
  [0, 0, 0, 1],
  [0, 1, 1, 0],
  [1, 0, 0, 0]
])
stick_3 = np.zeros((7,7))
stick_3[:4,3:] = np.array([
  [0, 0, 0, 1],
  [0, 0, 1, 0],
  [0, 1, 0, 0],
  [1, 0, 0, 0]
])
stick_4 = np.zeros((7,7))
stick_4[:4,3:] = np.array([
  [0, 0, 1, 0],
  [0, 1, 0, 0],
  [0, 1, 0, 0],
  [1, 0, 0, 0]
])
stick_5 = np.zeros((7,7))
stick_5[:4,3:] = np.array([
  [0, 1, 0, 0],
  [0, 1, 0, 0],
  [1, 0, 0, 0],
  [1, 0, 0, 0]
])
stick_full = np.zeros((24,7,7))
gaussian_kernel = get_gaussian_kernel(l=7, sig=3, normalize=False)
for i in range(0,4):
  j = i*6
  stick_full[j+0] = stick_0
  stick_full[j+1] = stick_1
  stick_full[j+2] = stick_2
  stick_full[j+3] = stick_3
  stick_full[j+4] = stick_4
  stick_full[j+5] = stick_5
  stick_0 = np.rot90(stick_0)
  stick_1 = np.rot90(stick_1)
  stick_2 = np.rot90(stick_2)
  stick_3 = np.rot90(stick_3)
  stick_4 = np.rot90(stick_4)
  stick_5 = np.rot90(stick_5)

kernel_dx = 3
kernel_dy = 3
sigma = 1
sigma_square = sigma**2
### method1: for loops only
# img_transformed = np.zeros((img_height, img_width))
# for x in range(kernel_dy, img_height - kernel_dy):
#   for y in range(kernel_dx, img_width - kernel_dx):
#     output_pixel = 0
#     weight = 0
#     for i in range(0,24):
#       convolved_with_stick = stick_full[i, :, :] * img[x - kernel_dx: x + kernel_dx + 1, y - kernel_dy: y + kernel_dy + 1]
#       mean_local = (1/4) * np.sum(convolved_with_stick)
#       variance_local = (1/4) * np.sum(np.square(convolved_with_stick - mean_local))
#       g = math.e**(-variance_local/sigma_square)
#       output_pixel += (g*mean_local)
#       weight += g
#     output_pixel = output_pixel/(weight+0.00001)
#     img_transformed[x,y] = output_pixel
### method2: x and y: for loop. inner part: vectors
# img_transformed = np.zeros((img_height, img_width))
# for x in range(kernel_dy, img_height - kernel_dy):
#   for y in range(kernel_dx, img_width - kernel_dx):
#     convolved_with_stick_all = stick_full * img[x - kernel_dx: x + kernel_dx + 1, y - kernel_dy: y + kernel_dy + 1]
#     mean_all = (1/4) * np.sum(convolved_with_stick_all, axis=(1,2))
#     variance_all = (1/4) * np.sum(np.square(convolved_with_stick_all - mean_all[:, np.newaxis, np.newaxis]), axis=(1,2))
#     g_all = np.exp(-variance_all/sigma_square)
#     img_transformed[x,y] = (g_all*mean_all).sum()/(g_all.sum()+0.0001)
### method3: vectors only
img_transformed = np.zeros((img_height - 2*kernel_dx, img_width - 2*kernel_dy))
weight = 0
for i in range(stick_full.shape[0]):
  convolved_with_stick = img_sliding_window * stick_full[i]
  ### 1
  # mean = (1/4) * np.sum(convolved_with_stick, axis=(2,3))
  # variance = (1/4) * np.sum(np.square(convolved_with_stick - mean[:, :, np.newaxis, np.newaxis]), axis=(2,3))
  ### 2
  mean = (1/4) * np.sum(convolved_with_stick, axis=(2,3))
  mean_stick_product = mean[:,:,np.newaxis,np.newaxis]*stick_full[i]
  variance = (1/4) * np.sum(np.square(convolved_with_stick - mean_stick_product), axis=(2,3))
  g = np.exp(-variance/sigma_square)
  weight += g
  img_transformed += (g*mean)
img_transformed = weight*img_transformed

end = time.time()
print(end - start)

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(img_transformed, cmap='gray')
plt.show()
