from skimage import morphology
import numpy as np
import matplotlib.pyplot as plt

from utils.file.dicom_reader import read_dicom
from skimage.filters import frangi, hessian

### read image
img_stack = read_dicom('./data/23_kep_test/Prostate/US_9660096_24_1.IMA')
img_dva = np.std(img_stack, axis=2, ddof=1)


### morphology
disk = morphology.disk(radius=20)
img_dva_scaled = (-1)*img_dva + abs(np.amin(img_dva))
img_morphological_closing = morphology.erosion(morphology.dilation(img_dva_scaled, disk))
img_diff1  = img_dva_scaled - img_morphological_closing

disk = morphology.disk(radius=20)
img_morphological_opening = morphology.dilation(morphology.erosion(img_dva_scaled, disk))
img_diff2 = img_dva_scaled - img_morphological_opening

### frangi 
# img_frangi = frangi(img_dva, black_ridges=True)


plt.figure(figsize=(12,7))
plt.subplot(1,3,1)
plt.imshow(img_dva, cmap='gray')
plt.subplot(1,3,2)
plt.imshow(img_diff1, cmap='gray')
plt.subplot(1,3,3)
plt.imshow(img_diff2, cmap='gray')
plt.show()

# np.savetxt('./outputs/pca/img_dva' + '.txt', img_stack_morph, delimiter='\t')
# np.savetxt('./outputs/pca/img_dsa' + '.txt', img_dsa, delimiter='\t')