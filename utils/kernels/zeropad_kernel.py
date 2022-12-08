import numpy as np

def get_zeropadded_kernel(img, kernel):
  # Zero-pad the kernel so same size as img
  h, w = img.shape
  kh, kw = kernel.shape
  kernel_padded = np.zeros_like(img, dtype="float64")
  kernel_padded[h//2-kh//2:h//2+kh//2+1, w//2-kw//2:w//2+kw//2+1] = kernel
  kernel_padded = np.fft.ifftshift(kernel_padded)
  return kernel_padded