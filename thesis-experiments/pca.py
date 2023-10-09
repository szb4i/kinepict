import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# setting path
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utils.file.dicom_reader import read_dicom
from skimage.filters import frangi
from skimage import morphology
from sklearn.decomposition import PCA


def get_pca_filtered_image(img, threshold_variance_ratio):
  pca = PCA(n_components=img.shape[1])
  transformed = pca.fit_transform(img)
  transformed_copy = transformed.copy()
  for i, variance_ratio in enumerate(pca.explained_variance_ratio_):
    if variance_ratio < threshold_variance_ratio:
      transformed_copy[:, i:] = 0
      break
  img_pca_filtered = pca.inverse_transform(transformed_copy)
  return img_pca_filtered
  

img_stack = read_dicom('./data/23_kep_test/Prostate/US_9660096_24_1.IMA')
# img_stack = read_dicom('./data/23_kep_test/Vena I/1_1_N')
img_dva = np.std(img_stack, axis=2, ddof=1)

# frangi_img, frangi_scale, frangi_direction = FrangiFilter2D(img, BlackWhite=False)
# https://scikit-image.org/docs/0.14.x/api/skimage.filters.html#skimage.filters.frangi
img_frangi = frangi(img_dva, black_ridges=True)

### opening: eliminates single pixels (noise removal)
square = morphology.square(width=2)
img_morphological_opening = morphology.dilation(morphology.erosion(img_frangi, square))
img_morphological_opening[0.15 > img_morphological_opening] = 0
img_morphological_opening[0.15 < img_morphological_opening] = 1
img_morphological_opening = img_morphological_opening.astype(bool)

img_stack_flatten = np.reshape(img_stack, (img_stack.shape[0]*img_stack.shape[1], img_stack.shape[2]))
img_pca_filtered_vein_domain = get_pca_filtered_image(img_stack_flatten[img_morphological_opening.flatten()], 1e-05)
img_pca_filtered_background_domain = get_pca_filtered_image(img_stack_flatten[~img_morphological_opening.flatten()], 5e-05)

img_stack_filtered = np.zeros((img_stack.shape[0], img_stack.shape[1], img_stack.shape[2]))
for k in range(0, img_stack.shape[2]):
  index_vein_domain = 0
  index_background_domain = 0
  for i in range(0, img_stack.shape[0]):
    for j in range(0, img_stack.shape[1]):
      if img_morphological_opening[i,j]:
        img_stack_filtered[i,j,k] = img_pca_filtered_vein_domain[index_vein_domain, k]
        index_vein_domain += 1
      else:
        img_stack_filtered[i,j,k] = img_pca_filtered_background_domain[index_background_domain, k]
        index_background_domain += 1

plt.figure(figsize=(12,7))
plt.subplot(1,2,1)
plt.axis("off")
plt.imshow(img_dva, cmap='gray')
plt.subplot(1,2,2)
plt.axis("off")
plt.imshow(np.std(img_stack_filtered, axis=2, ddof=1), cmap='gray')
plt.tight_layout()
plt.show()
