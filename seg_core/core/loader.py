"""WSI 统一加载与组织掩膜"""
import os
import cv2
import numpy as np


def load_tissue_mask(mask_path, proc_shape_yx=None):
    """读取组织掩膜，返回二值 uint8；若 proc_shape_yx 给出则 resize 到该尺寸"""
    if mask_path and os.path.isfile(mask_path):
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        m = (m > 0).astype(np.uint8)
        if proc_shape_yx is not None and m.shape != tuple(proc_shape_yx):
            m = cv2.resize(m, (proc_shape_yx[1], proc_shape_yx[0]), interpolation=cv2.INTER_NEAREST)
        return m
    return None
