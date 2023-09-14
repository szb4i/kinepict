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

v = np.fft.fftfreq(img.shape[0])
u = np.fft.fftfreq(img.shape[1])
vv, uu = np.meshgrid(v, u, indexing='ij')


sigma_1 = [0.5]
sigma_2 = [0.55]
# # plt.figure(figsize=(12,7))
for i in range(len(sigma_1)):
    print(i)
    gauss_1 = np.exp(-(np.hypot(uu, vv))**2/(2*sigma_1[i]**2))
    gauss_2 = np.exp(-(np.hypot(uu, vv))**2/(2*sigma_2[i]**2))
    gauss_diff = gauss_2 - gauss_1
    img_filtered_normal = np.fft.ifft2(img_f * gauss_diff).real
    np.savetxt('./outputs/final/normal' + str(i) +'.txt', img_filtered_normal, delimiter='\t')
    
### modified gauss diff
C0 = 0.25
W = 0.15
filter_f = np.exp(-(np.hypot(uu, vv) - C0)**2/(W**2))
### normal gauss diff
# sigma_1 = 0.2
# sigma_2 = 0.21
# gauss_1 = np.exp(-(np.hypot(uu, vv))**2/(2*sigma_1**2))
# gauss_2 = np.exp(-(np.hypot(uu, vv))**2/(2*sigma_2**2))
# gauss_diff = gauss_2 - gauss_1
# plt.figure(figsize=(12,7))
# plt.subplot(1,2,1)
# plt.imshow(np.fft.ifftshift(gauss_diff), cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(np.fft.ifftshift(filter_f), cmap='gray')
# plt.show()
img_filtered = np.fft.ifft2(img_f * filter_f).real
img_sum = img + 10*img_filtered
np.savetxt('./outputs/final/sum0.txt', img_filtered, delimiter='\t')




# sigma_1 = 0.19
# sigma_2 = 0.2
# gauss_1 = np.exp(-(np.hypot(uu, vv))**2/(2*sigma_1**2))
# gauss_2 = np.exp(-(np.hypot(uu, vv))**2/(2*sigma_2**2))
# gauss_diff = gauss_2 - gauss_1
# img_filtered_c = np.fft.ifft2(img_f * gauss_diff).real
# plt.figure(figsize=(12,7))
# plt.subplot(1,2,1)
# plt.imshow(img, cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(img + 50*img_filtered_c, cmap='gray')
# plt.show()

# time_start = time.time()
# time_end = time.time()
# print(time_end-time_start)

# np.savetxt('./outputs/algo2/sharpen1-prostate.txt', img, delimiter='\t')