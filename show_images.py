import os
import matplotlib.pyplot as plt
import math
import pydicom
import numpy as np
from skimage import exposure

from utils.file.dva_reader import read_dva
from src.methods.method2 import apply_method2
from src.methods.method3 import apply_method3
from src.methods.method4 import apply_method4
from src.methods.method5 import apply_method5
from src.methods.pca_method import apply_pca_method
from utils.file.dicom_reader import read_dicom
from utils.img.scaler import scale


img_dir = '/data/23_kep_test/'

### save transformed images
# apply_method = apply_method2
# method_name = 'method2'
# current_path = (os.path.abspath(os.path.dirname(__file__))).replace("\\","/")
# img_dir_list = os.listdir(current_path + img_dir)
# for i, dir_name in enumerate(img_dir_list):
#   inner_img_dir = current_path + img_dir + dir_name
#   image_list = os.listdir(inner_img_dir)
#   for j, img_file_name in enumerate(image_list):
#     if '.txt' == img_file_name[-4:]:
#       continue
#     img = read_dva(inner_img_dir + '/' + img_file_name)
#     ### save transformed dva
#     img_transformed = apply_method(img)
#     img_transformed_file_name = (img_file_name[:-4] + '_' + method_name + '.txt') if ('.dcm' == img_file_name[-4:] or '.IMA' == img_file_name[-4:]) else (img_file_name + '_' + method_name + '.txt')
#     np.savetxt(inner_img_dir + '/' + img_transformed_file_name, img_transformed, delimiter='\t')
#     print('saved ' + inner_img_dir + '/' + img_transformed_file_name)
#     ### save original dva
#     # img_transformed_file_name = (img_file_name[:-4] + '_' + 'dva' + '.txt') if ('.dcm' == img_file_name[-4:] or '.IMA' == img_file_name[-4:]) else (img_file_name + '_' + 'dva' + '.txt')
#     # np.savetxt(inner_img_dir + '/' + img_transformed_file_name, img, delimiter='\t')
#     # print('saved ' + inner_img_dir + '/' + img_transformed_file_name)

### show saved images
# current_path = (os.path.abspath(os.path.dirname(__file__))).replace("\\","/")
# img_dir_list = os.listdir(current_path + img_dir)
# for i, dir_name in enumerate(img_dir_list):
#   inner_img_dir = current_path + img_dir + dir_name
#   image_list = os.listdir(inner_img_dir)
#   for j, img_file_name in enumerate(image_list):
#     if '.txt' == img_file_name[-4:]:
#       continue
#     dva_file_name = (img_file_name[:-4] + '_' + 'dva' + '.txt') if ('.dcm' == img_file_name[-4:] or '.IMA' == img_file_name[-4:]) else (img_file_name + '_' + 'dva' + '.txt')
#     method3_file_name = (img_file_name[:-4] + '_' + 'method3' + '.txt') if ('.dcm' == img_file_name[-4:] or '.IMA' == img_file_name[-4:]) else (img_file_name + '_' + 'method3' + '.txt')
#     method4_file_name = (img_file_name[:-4] + '_' + 'method4' + '.txt') if ('.dcm' == img_file_name[-4:] or '.IMA' == img_file_name[-4:]) else (img_file_name + '_' + 'method4' + '.txt')
#     method2_file_name = (img_file_name[:-4] + '_' + 'method2' + '.txt') if ('.dcm' == img_file_name[-4:] or '.IMA' == img_file_name[-4:]) else (img_file_name + '_' + 'method2' + '.txt')
#     img_dva = np.loadtxt(inner_img_dir + '/' + dva_file_name)
#     method3 = np.loadtxt(inner_img_dir + '/' + method3_file_name)
#     method4 = np.loadtxt(inner_img_dir + '/' + method4_file_name)
#     method2 = np.loadtxt(inner_img_dir + '/' + method2_file_name)
#     plt.figure(figsize=(16,8))
#     plt.suptitle(img_file_name, size=16)
#     plt.subplot(2,2,1)
#     plt.imshow(img_dva, cmap='gray')
#     plt.gca().set_title('img')
#     plt.subplot(2,2,2)
#     plt.imshow(method3, cmap='gray')
#     plt.gca().set_title('method3')
#     plt.subplot(2,2,3)
#     plt.imshow(method4, cmap='gray')
#     plt.gca().set_title('method4')
#     plt.subplot(2,2,4)
#     plt.imshow(method2, cmap='gray')
#     plt.gca().set_title('method2')
#     plt.show()

### run on stack
# current_path = (os.path.abspath(os.path.dirname(__file__))).replace("\\","/")
# img_dir_list = os.listdir(current_path + img_dir)
# for i, dir_name in enumerate(img_dir_list):
#   inner_img_dir = current_path + img_dir + dir_name
#   image_list = os.listdir(inner_img_dir)
#   for j, img_file_name in enumerate(image_list):
#     if '.txt' == img_file_name[-4:]:
#       continue
#     img_stack = read_dicom(inner_img_dir + '/' + img_file_name)
#     img_stack_filtered = apply_pca_method(img_stack)
#     plt.figure(figsize=(12,7))
#     plt.suptitle(img_file_name, size=16)
#     plt.subplot(1,2,1)
#     plt.imshow(np.std(img_stack, axis=2, ddof=1), cmap='gray')
#     plt.gca().set_title('dva')
#     plt.subplot(1,2,2)
#     plt.imshow(np.std(img_stack_filtered, axis=2, ddof=1), cmap='gray')
#     plt.gca().set_title('dva_filtered')
#     plt.show()

img = read_dicom((os.path.abspath(os.path.dirname(__file__))).replace("\\","/") + img_dir + 'Vena I/1_1_N')
plt.figure(figsize=(12,7))
plt.subplot(1,4,1)
plt.imshow(img[:,:,0], cmap='gray')
plt.gca().set_title('Frame 1')
plt.subplot(1,4,2)
plt.imshow(img[:,:,24], cmap='gray')
plt.gca().set_title('Frame 25')
plt.subplot(1,4,3)
plt.imshow(img[:,:,49], cmap='gray')
plt.gca().set_title('Frame 50')
plt.subplot(1,4,4)
plt.imshow(img[:,:,-1] - img[:,:,0] , cmap='gray')
plt.gca().set_title('DSA')
plt.tight_layout()
plt.show()

