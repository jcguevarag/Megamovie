# Import all necessary libraries. Skimage is shorthand for scikit-image
import skimage.io
import skimage.util
import skimage.color
import skimage.transform
import copy
# import cv2
import numpy as np
# import multiprocessing
from matplotlib import pyplot as plt
from configuration import Configuration
from datetime import datetime
import sunpy
import sunpy.map
# from joblib import Parallel, delayed



def analyze(c,b):
    THRESHOLD = 0.5  # The highest pixel value that will be considered as part of the corona
    REALLY_BIG = 10  # Placeholder value. Can be changed to anything above 1.
    ROTATE_BY = 5  # Angle rotation step. Change this to change the degrees you want to rotate it by.
    STORE_EVERY = 10  # Every STORE_EVERY configuration will be saved.
    configurations = None
    # img1_points =[int(x) for x in kp1[mat[0]][0]]
    # img2_points = [int(x) for x in kp2[mat[1]][0]]
    # offset_x = img2_points[0] - img1_points[0]
    # offset_y = img2_points[1] - img1_points[1]
    area_of = 0
    for j in range(0,360,ROTATE_BY): # Rotate up to 360 Degrees by the Angle Step
        print('We are on degree: {}'.format(str(j)), 'The Time is: {}'.format(str(datetime.now())))
        f = copy.deepcopy(c)
        e = skimage.transform.rotate(b, j, center=[3072, 3072], preserve_range=True)
        e = skimage.util.invert(e) # Prepare the image to be super-imposed on the constant image
        # smallest_row = None # The smallest row where a corona pixel is detected using THRESHOLD
        # smallest_col = (0,REALLY_BIG) # The smallest column where a corona pixel is detected using THRESHOLD
        # greatest_row = (0,0) # The greatest row where a corona pixel is detected using THRESHOLD
        # greatest_col = (0,0) # The greatest col where a corona pixel is detected using THRESHOLD

        # for x in range(len(f)): # For every row in the constant image
        #     for y in range(len(f[x])): # For every column in that row
        #         try:
        #             val_a = f[x][y]
        #         except IndexError:
        #             val_a = 1
        #         try:
        #             val_b = e[x][y]
        #         except IndexError:
        #             val_b = 1
        #         val = min(val_a, val_b)
        #         if val < THRESHOLD:
        #             area_of += 1
        #         f[x][y] = val # Record the super-imposed value into f
        f = np.minimum(f,e)
        area_of = (f < THRESHOLD).sum()
        config = Configuration(area_of,f,e,j) # Creates a new configuration
        if configurations is None or config < configurations:
            configurations = config
    return configurations
# if __name__ == '__main__':
# NUM_CORES = 6
orig_a = sunpy.map.Map('704f3955a40167b232b5c825f9bb1ff735dcabcdfffbc84d9dfad9b0e25e6f8d_level_0.fits').data
orig_b = sunpy.map.Map('8e6861a87b4ded5e7993c0bfa7ec850cb7b076064edd13d833b2444668d88942_level_0.fits').data

a = skimage.color.rgb2gray(orig_a)  # The static, already aligned image in grayscale
b = skimage.color.rgb2gray(orig_b)  # The modular image that we are trying to align in grayscale
#b = skimage.transform.rotate(b, 100, center=[3072,3072], preserve_range=True)
c = skimage.util.invert(a)  # Invert the image so that black becomes white, and vice-versa
# g = skimage.util.invert(b)
configurations = None  # Stores the best configuration

del(orig_a)
del(orig_b)

res = analyze(c,b)

    # def check_rotation(rotate):
    #     configurations = None
    #     for j in range(0,360,rotate): # Rotate up to 360 Degrees by the Angle Step
    #         print('We are on match: {} We are on degree: {}'.format(str(matches.index(mat) + 1),str(j)), 'The Time is: {}'.format(str(datetime.now())))
    #         f = copy.deepcopy(c)
    #         e = skimage.transform.rotate(b, j, center=img2_points, preserve_range=True)
    #         e = skimage.util.invert(e) # Prepare the image to be super-imposed on the constant image
    #         smallest_row = None # The smallest row where a corona pixel is detected using THRESHOLD
    #         smallest_col = (0,REALLY_BIG) # The smallest column where a corona pixel is detected using THRESHOLD
    #         greatest_row = (0,0) # The greatest row where a corona pixel is detected using THRESHOLD
    #         greatest_col = (0,0) # The greatest col where a corona pixel is detected using THRESHOLD
    #
    #         for x in range(len(f)): # For every row in the constant image
    #             for y in range(len(f[x])): # For every column in that row
    #                 try:
    #                     val_a = f[x][y]
    #                 except IndexError:
    #                     val_a = 1
    #                 try:
    #                     val_b = e[x + offset_x][y + offset_y]
    #                 except IndexError:
    #                     val_b = 1
    #                 val = min(val_a, val_b)
    #                 if smallest_row is None and val < THRESHOLD:
    #                     smallest_row = (x,y)
    #                 if val < THRESHOLD and y < smallest_col[1]:
    #                     smallest_col = (x,y)
    #                 if val < THRESHOLD:
    #                     greatest_row = (x,y)
    #                 if val < THRESHOLD and y > greatest_col[1]:
    #                     greatest_col = (x,y)
    #                 f[x][y] = val # Record the super-imposed value into f
    #         config = Configuration(smallest_row,smallest_col,greatest_row,greatest_col,f,e,j) # Creates a new configuration
    #         if configurations is None or config < configurations:
    #             configurations = config
    #         return configurations

    # stored = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(analyze)(mat,c,b) for mat in matches)
