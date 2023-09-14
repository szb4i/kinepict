import numpy as np
import matplotlib.pyplot as plt
from utils.file.dva_reader import read_dva
from utils.img.scaler import scale
import time

### read dva
img = read_dva('./data/23_kep_test/Carotis 100%/CAR17IM0')
# img = read_dva('./data/23_kep_test/Prostate/US_9660096_24_1.IMA')
img = scale(img)
img_f = np.fft.fft2(img)

time_start = time.time()
sigma_1 = 0.19
sigma_2 = 0.2
v = np.fft.fftfreq(img.shape[0])
u = np.fft.fftfreq(img.shape[1])
vv, uu = np.meshgrid(v, u, indexing='ij')
gauss_1 = np.exp(-(np.hypot(uu, vv))**2/(2*sigma_1**2))
gauss_2 = np.exp(-(np.hypot(uu, vv))**2/(2*sigma_2**2))
gauss_diff = gauss_2 - gauss_1
img_filtered_c = np.fft.ifft2(img_f * gauss_diff).real
time_end = time.time()
print(time_end-time_start)
plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(img + 50*img_filtered_c, cmap='gray')
plt.show()

np.savetxt('./outputs/algo2/sharpen1-prostate.txt', img, delimiter='\t')