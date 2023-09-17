import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
import sys
import os
 
# setting path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.file.dva_reader import read_dva
from utils.img.clahe import get_clahe_img
from utils.img.scaler import scale
from utils.img.roi import get_roi_prostate, get_roi_carotis
from utils.img.snr import get_snr

def get_wl_operatored_img(img):
    C = 9*np.identity(9)+np.full((9,9), -1)
    normalizing_const = 1/(9**2)
    kernel_dx = 1
    kernel_dy = 1
    img_wl = np.zeros((img.shape[0], img.shape[1]))
    for x in range(kernel_dy, img.shape[1] - kernel_dy):
        for y in range(kernel_dx, img.shape[0] - kernel_dx):
            region_flatten = img[x - kernel_dx: x + kernel_dx + 1, y - kernel_dy: y + kernel_dy + 1].flatten()
            img_wl[x,y] = normalizing_const*(region_flatten.dot(C.dot(region_flatten)))
    return img_wl

img_1 = read_dva('./data/23_kep_test/Prostate/US_9660096_24_1.IMA')
img_wl_1 = get_wl_operatored_img(img_1)
img_wl_1 = get_clahe_img(scale(img_wl_1))
snr_1 = get_snr(get_roi_prostate(img_wl_1))
print(snr_1)

img_2 = read_dva('./data/23_kep_test/Carotis 100%/CAR17IM0')
img_wl_2 = get_wl_operatored_img(img_2)
img_wl_2 = get_clahe_img(scale(img_wl_2))
snr_2 = get_snr(get_roi_carotis(img_wl_2))
print(snr_2)

plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
plt.imshow(img_wl_1, cmap='gray')
plt.axis("off")
plt.gca().set_title('Prostate', fontdict={'fontsize':18})
plt.subplot(1,2,2)
plt.imshow(img_wl_2, cmap='gray')
plt.axis("off")
plt.gca().set_title('Carotis', fontdict={'fontsize':18})
plt.tight_layout()
plt.show()

# np.savetxt('./outputs/wl-prostate.txt', img_wl_1, delimiter='\t')
# np.savetxt('./outputs/wl-carotis.txt', img_wl_2, delimiter='\t')
# img_wl_1_enhanced = plt.imread('./outputs/wl-prostate-enhanced.png')
# img_wl_2_enhanced = plt.imread('./outputs/wl-carotis-enhanced.png')
# plt.figure(figsize=(12,7))
# plt.subplot(1,2,1)
# plt.imshow(img_wl_1_enhanced, cmap='gray')
# plt.axis("off")
# plt.gca().set_title('Prostate', fontdict={'fontsize':18})
# plt.subplot(1,2,2)
# plt.imshow(img_wl_2_enhanced, cmap='gray')
# plt.axis("off")
# plt.gca().set_title('Carotis', fontdict={'fontsize':18})
# plt.tight_layout()
# plt.show()
