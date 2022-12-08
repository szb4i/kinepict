import numpy as np

def scale(img, max=1, min=0):
  old_range = img.max() - img.min()
  new_range = max - min
  return ((img - img.min()) * new_range/old_range) + min
