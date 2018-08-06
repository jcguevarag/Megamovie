import skimage.io
import skimage.util
import skimage.color
import skimage.transform
import os

from sort_by_bin import bin_sort, split_bin
from imgcomparev2 import analyze
from joblib import Parallel, delayed


def analyze_bin(section, areas_dict, section_num, BASE_DIRECTORY):
    base_image = areas_dict[section[len(section) // 2]] # Average area in this bin
    base_image = skimage.io.imread(BASE_DIRECTORY + base_image)
    base_image = skimage.color.rgb2gray(base_image)
    base_image = skimage.util.invert(base_image)
    os.makedirs('bin_' + str(section_num))
    for img in section:
        analyze(base_image,skimage.color.rgb2gray(skimage.io.imread(BASE_DIRECTORY + areas_dict[img])),areas_dict[img], 'bin_' + str(section_num))


if __name__ == '__main__':

    NUMBER_OF_BINS = 10
    BASE_DIRECTORY = 'images_random_200/'

    areas, areas_dict = bin_sort()
    nested_areas = split_bin(areas, NUMBER_OF_BINS)

    for x in range(len(nested_areas)):
        analyze_bin(nested_areas[x], areas_dict, x, BASE_DIRECTORY)

    #Parallel(n_jobs=2)(delayed(analyze_bin(nested_areas[x], areas_dict, x, BASE_DIRECTORY) for x in range(len(nested_areas))))


