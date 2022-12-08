import pydicom
import numpy as np

def read_dicom(file_path):
    with pydicom.read_file(file_path, force=True) as img:
        raw = np.moveaxis(img.pixel_array, 0, -1).astype(np.float64)
        return raw
