# 2) Szimmetrikus kernel
# a) Élkiemelés 3x3-as Laplace alapú szűrővel. 1x, 2x, 3x alkalmazva
# b) Simítás 3x3-as Gauss szűrővel. 1x, 2x, 3x alkalmazva
# c) Simítás 3x3-as box szűrővel. 1x, 2x, 3x alkalmazva
# d) Az a) és b), illetve az a) és c) kombinációja.
# megfigyelés: d) módszernél egy idő után ringing
import numpy as np
import matplotlib.pyplot as plt
from utils.file.dva_reader import read_dva
from scipy.signal import convolve2d
from utils.img.scaler import scale
from utils.kernels.gaussian_kernel import get_gaussian_kernel
from utils.kernels.box_kernel import get_box_kernel
import numpy as np
from skimage import exposure
from utils.filter.anisodiff import anisodiff

### read dva
img = read_dva('./data/23_kep_test/Prostate/US_9660096_24_1.IMA')
# img = read_dva('./data/23_kep_test/Carotis 100%/CAR17IM0')
img = scale(img)

### a)
# f_sharpen=np.array([
#     [-1,-1,-1],
#     [-1,12,-1],
#     [-1,-1,-1]
# ])
# img_sharpened_1 = convolve2d(img, f_sharpen, mode='same', boundary = 'symm', fillvalue=0)
# img_sharpened_2 = convolve2d(img_sharpened_1, f_sharpen, mode='same', boundary = 'symm', fillvalue=0)
# img_sharpened_3 = convolve2d(img_sharpened_2, f_sharpen, mode='same', boundary = 'symm', fillvalue=0)
# plt.figure(figsize=(12,7))
# plt.subplot(2,2,1)
# plt.imshow(img, cmap='gray')
# plt.subplot(2,2,2)
# plt.imshow(img_sharpened_1, cmap='gray')
# plt.subplot(2,2,3)
# plt.imshow(img_sharpened_2, cmap='gray')
# plt.subplot(2,2,4)
# plt.imshow(img_sharpened_3, cmap='gray')
# plt.show()

### b)
# kernel = get_gaussian_kernel()
# img_blured_1 = convolve2d(img, kernel, mode='same')
# img_blured_2 = convolve2d(img_blured_1, kernel, mode='same')
# img_blured_3 = convolve2d(img_blured_2, kernel, mode='same')
# plt.figure(figsize=(12,7))
# plt.imshow(img_blured_3, cmap='gray')
# plt.show()

### c)
# kernel = get_box_kernel()
# img_blured_1 = convolve2d(img, kernel, mode='same')
# img_blured_2 = convolve2d(img_blured_1, kernel, mode='same')
# img_blured_3 = convolve2d(img_blured_2, kernel, mode='same')
# plt.figure(figsize=(12,7))
# plt.subplot(2,2,1)
# plt.imshow(img, cmap='gray')
# plt.subplot(2,2,2)
# plt.imshow(img_blured_1, cmap='gray')
# plt.subplot(2,2,3)
# plt.imshow(img_blured_2, cmap='gray')
# plt.subplot(2,2,4)
# plt.imshow(img_blured_3, cmap='gray')
# plt.show()

### d)
img = read_dva('./data/23_kep_test/Prostate/US_9660096_24_1.IMA')
# img = read_dva('./data/23_kep_test/Carotis 100%/CAR17IM0')
img = scale(img)
kernel_sharpen=np.array([
    [-1,-1,-1],
    [-1,12,-1],
    [-1,-1,-1]
])
kernel_blur = get_gaussian_kernel(l=3)
for i in range(1,3):
    img = convolve2d(img, kernel_sharpen, mode='same', boundary = 'symm', fillvalue=0)
    img = convolve2d(img, kernel_blur, mode='same', boundary = 'symm', fillvalue=0)
    # img = scale(img)
    # img = exposure.equalize_adapthist(img,kernel_size=[100,100],clip_limit=0.0000025,nbins=26200)
plt.figure(figsize=(12,7))
plt.imshow(img, cmap='gray')
plt.show()

# np.savetxt('./outputs/algo2/sharpen1-prostate.txt', img, delimiter='\t')

# adaptive histogram
from skimage import exposure
img = scale(img)
img_clahe = exposure.equalize_adapthist(img,kernel_size=[100,100],clip_limit=0.0002,nbins=26200)
plt.imshow(img_clahe, cmap='gray')