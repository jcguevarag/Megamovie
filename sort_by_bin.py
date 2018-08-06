import os
import skimage.io
import skimage.transform
import skimage.util
import numpy as np


def bin_sort():
    THRESHOLD = 0.4
    directory = os.listdir(os.listdir('.')[7])
    areas = []
    areas_dict = {}
    print(directory)
    for img in directory:
        img_name = img
        img = skimage.io.imread('images_random_200/' + img, as_gray=True)
        img = skimage.util.invert(img)
        area = (img < THRESHOLD).sum()
        print(area)
        areas.append(area)
        areas_dict[area] = img_name

    return sorted(areas), areas_dict


def split_bin(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out