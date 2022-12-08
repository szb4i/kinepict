import numpy as np

def get_laplacian_kernel():
  # if center element is +8: add to original
  # if center element is -8: subtract from original
  # return np.array([[1,1,1],[1,-8,1],[1,1,1]], dtype=float)
  return np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=float)