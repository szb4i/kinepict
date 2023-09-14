import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import sys
import os
 
# setting path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.kernels.sobel_kernels import get_sobel_x_kernel, get_sobel_y_kernel
from utils.file.dva_reader import read_dva
from utils.img.clahe import get_clahe_img
from utils.img.scaler import scale
from utils.img.roi import get_roi_prostate, get_roi_carotis
from utils.img.snr import get_snr

kernel_x = get_sobel_x_kernel()
kernel_y = get_sobel_y_kernel()

img_1 = read_dva('./data/23_kep_test/Prostate/US_9660096_24_1.IMA')
gx = convolve2d(img_1, kernel_x, mode='same', boundary = 'symm', fillvalue=0)
gy = convolve2d(img_1, kernel_y, mode='same', boundary = 'symm', fillvalue=0)
img_sobel_1 = np.hypot(gx, gy)
img_sobel_1 = get_clahe_img(scale(img_sobel_1))

img_2 = read_dva('./data/23_kep_test/Carotis 100%/CAR17IM0')
gx = convolve2d(img_2, kernel_x, mode='same', boundary = 'symm', fillvalue=0)
gy = convolve2d(img_2, kernel_y, mode='same', boundary = 'symm', fillvalue=0)
img_sobel_2 = np.hypot(gx, gy)
img_sobel_2 = get_clahe_img(scale(img_sobel_2))

snr_1 = get_snr(get_roi_prostate(img_sobel_1))
print(snr_1)

snr_2 = get_snr(get_roi_carotis(img_sobel_2))
print(snr_2)

plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
plt.imshow(img_sobel_1, cmap='gray')
plt.axis("off")
plt.gca().set_title('Prostate', fontdict={'fontsize':18})
plt.subplot(1,2,2)
plt.imshow(img_sobel_2, cmap='gray')
plt.axis("off")
plt.gca().set_title('Carotis', fontdict={'fontsize':18})
plt.tight_layout()
plt.show()

