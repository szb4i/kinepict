from utils.file.dicom_reader import read_dicom
import numpy as np

def read_dva(file_path):
  ima = read_dicom(file_path)
  dva_img = np.std(ima, axis=2, ddof=1)
  return dva_img
