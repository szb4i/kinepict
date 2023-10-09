import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# setting path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.file.dva_reader import read_dva
from utils.img.clahe import get_clahe_img
from utils.img.scaler import scale
from utils.img.roi import get_roi_prostate, get_roi_carotis
from utils.img.snr import get_snr

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# setting path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.file.dva_reader import read_dva
from utils.img.clahe import get_clahe_img
from utils.img.scaler import scale
from utils.img.roi import get_roi_prostate, get_roi_carotis
from utils.img.snr import get_snr

def get_final_img(img):
    gauss_width = 0.15
    gauss_center = 0.25
    sum_weight = 5
    img = scale(img)
    img_fourier = np.fft.fft2(img)
    v = np.fft.fftfreq(img.shape[0])
    u = np.fft.fftfreq(img.shape[1])
    vv, uu = np.meshgrid(v, u, indexing='ij')
    filter_fourier = np.exp(-(np.hypot(uu, vv) - gauss_center)**2/(gauss_width**2))
    img_filtered = np.fft.ifft2(img_fourier * filter_fourier).real
    img_final = img + sum_weight*img_filtered
    return img_final

img_1 = read_dva('./data/23_kep_test/Prostate/US_9660096_24_1.IMA')
img_final_1 = get_final_img(img_1)
img_final_1 = get_clahe_img(scale(img_final_1))

img_2 = read_dva('./data/23_kep_test/Carotis 100%/CAR17IM0')
img_final_2 = get_final_img(img_2)
img_final_2 = get_clahe_img(scale(img_final_2))

snr_1 = get_snr(get_roi_prostate(img_final_1))
print(snr_1)

snr_2 = get_snr(get_roi_carotis(img_final_2))
print(snr_2)

plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
plt.imshow(img_final_1, cmap='gray')
plt.axis("off")
plt.gca().set_title('Prostate', fontdict={'fontsize':18})
plt.subplot(1,2,2)
plt.imshow(img_final_2, cmap='gray')
plt.axis("off")
plt.gca().set_title('Carotis', fontdict={'fontsize':18})
plt.tight_layout()
plt.show()
