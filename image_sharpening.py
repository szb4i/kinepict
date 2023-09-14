import numpy as np
import matplotlib.pyplot as plt
from utils.file.dva_reader import read_dva
from scipy.signal import convolve2d
from utils.img.scaler import scale
from scipy.ndimage import gaussian_filter
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from utils.kernels.gaussian_kernel import get_gaussian_kernel
import numpy as np

img = read_dva('./data/23_kep_test/Prostate/US_9660096_24_1.IMA')
img = scale(img)

f_sharpen=np.array([
    [-1,-1,-1],
    [-1,9,-1],
    [-1,-1,-1]
])
img_sharpened = convolve2d(img, f_sharpen, mode='same', boundary = 'symm', fillvalue=0)

plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(img_sharpened, cmap='gray')
plt.show()

### save images as text
# np.savetxt('./outputs/image_sharpening/img.txt', img, delimiter='\t')
# np.savetxt('./outputs/image_sharpening/img_sharpened.txt', img_sharpened, delimiter='\t')