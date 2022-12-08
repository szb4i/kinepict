import numpy as np

def get_sobel_x_kernel():
  return np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float)

def get_sobel_y_kernel():
  return np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float)