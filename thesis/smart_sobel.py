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

def get_smart_sobel_img(img):
    kernel_x = get_sobel_x_kernel()
    kernel_y = get_sobel_y_kernel()
    img_gradient_x = convolve2d(img, kernel_x, mode='same', boundary = 'symm', fillvalue=0)
    img_gradient_y = convolve2d(img, kernel_y, mode='same', boundary = 'symm', fillvalue=0)
    img_gradient_direction = np.arctan2(img_gradient_y, img_gradient_x)
    img_gradient_magnitude = scale(np.hypot(img_gradient_x, img_gradient_y))
    img_gradient_direction_gradient_x = convolve2d(img_gradient_direction, kernel_x, mode='same', boundary = 'symm', fillvalue=0)
    img_gradient_direction_gradient_y = convolve2d(img_gradient_direction, kernel_y, mode='same', boundary = 'symm', fillvalue=0)
    img_gradient_direction_gradient_mangitude = np.hypot(img_gradient_direction_gradient_x, img_gradient_direction_gradient_y)
    # inversion: where directional change is small -> veins; where directional change is big -> noise. inverting it to have strong signal where veins are
    img_gradient_direction_gradient_mangitude_inverted = np.log(1/(img_gradient_direction_gradient_mangitude+0.0001))
    img_gradient_direction_gradient_mangitude_inverted = scale(img_gradient_direction_gradient_mangitude_inverted)
    return img_gradient_direction_gradient_mangitude_inverted

img_1 = read_dva('./data/23_kep_test/Prostate/US_9660096_24_1.IMA')
img_smart_sobel_1 = get_smart_sobel_img(img_1)
img_smart_sobel_1 = get_clahe_img(scale(img_smart_sobel_1))

img_2 = read_dva('./data/23_kep_test/Carotis 100%/CAR17IM0')
img_smart_sobel_2 = get_smart_sobel_img(img_2)
img_smart_sobel_2 = get_clahe_img(scale(img_smart_sobel_2))

snr_1 = get_snr(get_roi_prostate(img_smart_sobel_1))
print(snr_1)

snr_2 = get_snr(get_roi_carotis(img_smart_sobel_2))
print(snr_2)

plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
plt.imshow(img_smart_sobel_1, cmap='gray')
plt.axis("off")
plt.gca().set_title('Prostate', fontdict={'fontsize':18})
plt.subplot(1,2,2)
plt.imshow(img_smart_sobel_2, cmap='gray')
plt.axis("off")
plt.gca().set_title('Carotis', fontdict={'fontsize':18})
plt.tight_layout()
plt.show()