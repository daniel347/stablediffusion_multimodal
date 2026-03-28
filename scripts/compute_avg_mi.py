import cv2
import torch
import numpy as np
from pathlib import Path
from sklearn.neighbors import KernelDensity
import numpy as np

def compute_mutual_information(colour_img, semseg_img):
    colour_img = colour_img.reshape(-1, 3)
    semseg_img = semseg_img.reshape(-1, 3)

    kde_colour = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(colour_img)
    kde_semseg = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(semseg_img)

    kde_joint = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(np.hstack((colour_img, semseg_img)))

    p_colour = kde_colour.score_samples(colour_img)
    p_semseg = kde_semseg.score_samples(semseg_img)
    p_joint = kde_joint.score_samples(np.hstack((colour_img, semseg_img)))

    H_c = np.mean(- p_colour)
    H_c_s = np.mean(- p_joint + p_semseg)

    MI = H_c - H_c_s
    return MI



if __name__ == "__main__":
    gen_path = Path("C:/Users/danie/OneDrive - Imperial College London/CAMSAP/2025_07_02__16_33_18_nyu_lsun_ae_diff_test_metrics")

    colour_img = list((gen_path / "temp_fid_colour").glob("*.png"))[0]
    semseg_img = list((gen_path / "temp_fid_depth").glob("*.png"))[0]

    colour_img = cv2.imread(str(colour_img))
    semseg_img = cv2.imread(str(semseg_img))

    print("MI unshifted:", compute_mutual_information(colour_img, semseg_img))

    semseg_img_shifted = np.roll(semseg_img, shift=1, axis=0)
    print("MI shifted:", compute_mutual_information(colour_img, semseg_img_shifted))
