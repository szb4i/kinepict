import numpy as np

def get_kirsch_kernel_0():
  return np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]])

def get_kirsch_kernel_45():
  return np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]])

def get_kirsch_kernel_90():
  return np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]])

def get_kirsch_kernel_135():
  return np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]])

def get_kirsch_kernel_180():
  return np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]])

def get_kirsch_kernel_225():
  return np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])

def get_kirsch_kernel_270():
  return np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]])

def get_kirsch_kernel_315():
  return np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])