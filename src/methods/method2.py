import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from PIL import Image as im

from utils.kernels.gaussian_kernel import get_gaussian_kernel
from utils.kernels.sobel_kernels import get_sobel_x_kernel, get_sobel_y_kernel
from utils.kernels.laplacian_kernel import get_laplacian_kernel
from utils.kernels.box_kernel import get_box_kernel
from utils.patterns.zone_plate_pattern import get_zone_plate_pattern
from utils.file.dicom_reader import read_dicom
from utils.file.dva_reader import read_dva
from utils.img.scaler import scale
from src.methods.method4 import apply_method4

def apply_method2(img):
  ### 2nd derivative
  kernel_laplacian = get_laplacian_kernel()
  img_laplacian = convolve2d(img, kernel_laplacian, mode='same')
  img_laplacian[img_laplacian < 0] = 0
  img_laplacian_sum = img + img_laplacian
  img_1stderivative = apply_method4(img)
  kernel_box = get_box_kernel(5)
  img_1stderivative_smooth = convolve2d(img_1stderivative, kernel_box, mode='same', boundary = 'symm', fillvalue=0)
  product = img_laplacian_sum * img_1stderivative_smooth
  img_product_sum = img + product
  img_product_sum_gamma = np.array(img_product_sum ** 0.5)
  return img_product_sum_gamma
