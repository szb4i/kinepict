import os
import matplotlib.pyplot as plt
import math
import pydicom
import numpy as np

from utils.file.dva_reader import read_dva
from src.methods.method3 import apply_method3
from src.methods.method4 import apply_method4
from src.methods.method5 import apply_method5
from utils.file.dicom_reader import read_dicom

img_dir = '/data/23_kep_test/'

### save transformed images
# apply_method = apply_method5
# method_name = 'method5'
# current_path = (os.path.abspath(os.path.dirname(__file__))).replace("\\","/")
# img_dir_list = os.listdir(current_path + img_dir)
# for i, dir_name in enumerate(img_dir_list):
#   inner_img_dir = current_path + img_dir + dir_name
#   image_list = os.listdir(inner_img_dir)
#   for j, img_file_name in enumerate(image_list):
#     if '.txt' == img_file_name[-4:]:
#       continue
#     img = read_dva(inner_img_dir + '/' + img_file_name)
#     img_transformed = apply_method(img)
#     img_transformed_file_name = (img_file_name[:-4] + '_' + method_name + '.txt') if ('.dcm' == img_file_name[-4:] or '.IMA' == img_file_name[-4:]) else (img_file_name + '_' + method_name + '.txt')
#     np.savetxt(inner_img_dir + '/' + img_transformed_file_name, img_transformed, delimiter='\t')
#     print('saved ' + inner_img_dir + '/' + img_transformed_file_name)

### show transformed images
current_path = (os.path.abspath(os.path.dirname(__file__))).replace("\\","/")
img_dir_list = os.listdir(current_path + img_dir)
for i, dir_name in enumerate(img_dir_list):
  inner_img_dir = current_path + img_dir + dir_name
  image_list = os.listdir(inner_img_dir)
  for j, img_file_name in enumerate(image_list):
    if '.txt' == img_file_name[-4:]:
      continue
    img = read_dva(inner_img_dir + '/' + img_file_name)
    method3_file_name = (img_file_name[:-4] + '_' + 'method3' + '.txt') if ('.dcm' == img_file_name[-4:] or '.IMA' == img_file_name[-4:]) else (img_file_name + '_' + 'method3' + '.txt')
    method4_file_name = (img_file_name[:-4] + '_' + 'method4' + '.txt') if ('.dcm' == img_file_name[-4:] or '.IMA' == img_file_name[-4:]) else (img_file_name + '_' + 'method4' + '.txt')
    method5_file_name = (img_file_name[:-4] + '_' + 'method5' + '.txt') if ('.dcm' == img_file_name[-4:] or '.IMA' == img_file_name[-4:]) else (img_file_name + '_' + 'method5' + '.txt')
    method3 = np.loadtxt(inner_img_dir + '/' + method3_file_name)
    method4 = np.loadtxt(inner_img_dir + '/' + method4_file_name)
    method5 = np.loadtxt(inner_img_dir + '/' + method5_file_name)
    plt.figure(figsize=(16,8))
    plt.suptitle(img_file_name, size=16)
    plt.subplot(2,2,1)
    plt.imshow(img, cmap='gray')
    plt.gca().set_title('img')
    plt.subplot(2,2,2)
    plt.imshow(method3, cmap='gray')
    plt.gca().set_title('method3')
    plt.subplot(2,2,3)
    plt.imshow(method4, cmap='gray')
    plt.gca().set_title('method4')
    plt.subplot(2,2,4)
    plt.imshow(method5, cmap='gray')
    plt.gca().set_title('method5')
    plt.show()
