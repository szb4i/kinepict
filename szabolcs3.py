# 3) Aszimmetrikus kernel
# a) Élkiemelés 3x3-as Laplace alapú szűrővel. 1x, 2x, 3x alkalmazva
# b) Simítás anisotropic diffusion filtert használva. 1x, 2x, 3x
# c) Az a) és b), illetve a b) és a) egymás után.
import numpy as np
import matplotlib.pyplot as plt
from utils.file.dva_reader import read_dva
from scipy.signal import convolve2d
from utils.img.scaler import scale
from utils.kernels.gaussian_kernel import get_gaussian_kernel
from utils.kernels.box_kernel import get_box_kernel
from utils.filter.anisodiff import anisodiff
import numpy as np
import scipy
from scipy import ndimage
from skimage import exposure


### read dva
img = read_dva('./data/23_kep_test/Prostate/US_9660096_24_1.IMA')
img = scale(img)

### a)
# f_sharpen=np.array([
#     [-1,-1,-1],
#     [-1,9,-1],
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
### https://github.com/awangenh/fastaniso
# img_diff = anisodiff(img)
# img_diff = anisodiff(img_diff)
# img_diff = anisodiff(img_diff)
# img_diff = anisodiff(img_diff)
# img_diff = anisodiff(img_diff)
# img_diff = anisodiff(img_diff)
# plt.figure(figsize=(12,7))
# plt.subplot(1,2,1)
# plt.imshow(img, cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(img_diff, cmap='gray')
# plt.show()

### c)
# while True:
#     img = convolve2d(img, f_sharpen, mode='same', boundary = 'symm', fillvalue=0)
#     img = anisodiff(img, niter=5)
#     plt.figure(figsize=(12,7))
#     plt.imshow(img, cmap='gray')
#     plt.show()

### imagej does something differently... 
f_sharpen=np.array([
    [-1,-1,-1],
    [-1,8,-1],
    [-1,-1,-1]
])
def sharpen(img_input):    
    img_sharp = convolve2d(img_input, f_sharpen, mode='same', boundary = 'symm', fillvalue=0)
    img_sharp[img_sharp<0] = 0
    img_sharp = img_input + img_sharp
    img_sharp[img_sharp>1] = 1
    return img_sharp



plt.figure(figsize=(12,7))
plt.imshow(img, cmap='gray')
plt.show()
img_sharp = img
img_sharp = sharpen(img_sharp)
img_sharp = anisodiff(img_sharp)
img_clahe = exposure.equalize_adapthist(img_sharp,kernel_size=[90,90],clip_limit=0.00015,nbins=26200)
plt.imshow(img_clahe, cmap='gray')
print('a')

### d) clahe on original
img_clahe = exposure.equalize_adapthist(img,kernel_size=[90,90],clip_limit=0.00015,nbins=26200)



# np.savetxt('./outputs/szabolcs3/img.txt', img, delimiter='\t')
# np.savetxt('./outputs/szabolcs3/img_diff.txt', img_diff, delimiter='\t')
