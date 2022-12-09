import os
import matplotlib.pyplot as plt
import math

from utils.file.dva_reader import read_dva
from src.methods.method3 import apply_method3
from src.methods.method4 import apply_method4
from src.methods.method5 import apply_method5


current_path = (os.path.abspath(os.path.dirname(__file__))).replace("\\","/")
images_path = current_path + '/data/X-ray 70%/'
dir_list = os.listdir(images_path)
number_of_columns = 4
number_of_rows = math.ceil(len(dir_list)/4)
for i, file_name in enumerate(dir_list):
  img = read_dva(images_path + file_name)
  img_transformed = apply_method4(img)
  plt.figure(figsize=(12,7))
  plt.subplot(1,2,1)
  plt.title(file_name)
  plt.imshow(img, cmap='gray')
  plt.subplot(1,2,2)
  plt.title(file_name + ' transformed')
  plt.imshow(img_transformed, cmap='gray')
  plt.show()
plt.show()
