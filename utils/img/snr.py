import numpy as np

def get_snr(img_roi_pairs):
    n = 0
    total_signal = 0
    total_area = 0
    for img_roi_pair in img_roi_pairs:
        signal = np.mean(img_roi_pair[0]) - np.mean(img_roi_pair[1])
        noise = np.std(img_roi_pair[1])
        area = img_roi_pair[0].size
        n += 1
        total_signal += (area*(signal/noise))
        total_area += area
    return total_signal/total_area