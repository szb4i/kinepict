
import numpy as np

from utils.file.dva_reader import read_dva
from utils.frangi.frangiFilter2D import FrangiFilter2D
from utils.file.dicom_reader import read_dicom
from skimage.filters import frangi, hessian
from skimage import morphology
from sklearn.decomposition import PCA
from utils.img.scaler import scale

def apply_pca_method(img_stack):
    img_dsa = np.amin(img_stack, axis=2) - np.amax(img_stack, axis=2)

    img_frangi = frangi(img_dsa, black_ridges=True)

    square = morphology.square(width=2)
    img_morphological_opening = morphology.dilation(morphology.erosion(img_frangi, square))
    img_morphological_opening[0.15 > img_morphological_opening] = 0
    img_morphological_opening[0.15 < img_morphological_opening] = 1
    img_morphological_opening = img_morphological_opening.astype(bool)

    img_stack_flatten = np.reshape(img_stack, (img_stack.shape[0]*img_stack.shape[1], img_stack.shape[2]))

    img_stack_flatten_vein_domain = img_stack_flatten[img_morphological_opening.flatten()]
    threshold_variance_ratio = 1e-05
    pca = PCA(n_components=img_stack_flatten_vein_domain.shape[1])
    transformed = pca.fit_transform(img_stack_flatten_vein_domain)
    transformed_copy = transformed.copy()
    for i, variance_ratio in enumerate(pca.explained_variance_ratio_):
    # if variance_ratio < threshold_variance_ratio:
        if i > 2:
            transformed_copy[:, i:] = 0
            break
    img_pca_filtered_vein_domain = pca.inverse_transform(transformed_copy)

    img_stack_flatten_background_domain = img_stack_flatten[~img_morphological_opening.flatten()]
    threshold_variance_ratio = 5e-05
    pca = PCA(n_components=img_stack_flatten_background_domain.shape[1])
    transformed = pca.fit_transform(img_stack_flatten_background_domain)
    transformed_copy = transformed.copy()
    for i, variance_ratio in enumerate(pca.explained_variance_ratio_):
    # if variance_ratio < threshold_variance_ratio:
        if i > 2:
            transformed_copy[:, i:] = 0
            break
    img_pca_filtered_background_domain = pca.inverse_transform(transformed_copy)

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
    return img_stack_filtered