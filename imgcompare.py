# Import all necessary libraries. Skimage is shorthand for scikit-image
import skimage.io
import skimage.util
import skimage.color
import skimage.transform
import copy
import cv2
import numpy as np
import multiprocessing
from matplotlib import pyplot as plt
from configuration import Configuration
from datetime import datetime
from joblib import Parallel, delayed



def analyze(mat,c,b,kp1,kp2):
    THRESHOLD = 0.2  # The highest pixel value that will be considered as part of the corona
    REALLY_BIG = 10  # Placeholder value. Can be changed to anything above 1.
    ROTATE_BY = 90  # Angle rotation step. Change this to change the degrees you want to rotate it by.
    STORE_EVERY = 10  # Every STORE_EVERY configuration will be saved.
    configurations = None
    img1_points =[int(x) for x in kp1[mat[0]][0]]
    img2_points = [int(x) for x in kp2[mat[1]][0]]
    offset_x = img2_points[0] - img1_points[0]
    offset_y = img2_points[1] - img1_points[1]
    for j in range(0,360,ROTATE_BY): # Rotate up to 360 Degrees by the Angle Step
        print('We are on match: {} We are on degree: {}'.format(str(matches.index(mat) + 1),str(j)), 'The Time is: {}'.format(str(datetime.now())))
        f = copy.deepcopy(c)
        e = skimage.transform.rotate(b, j, center=img2_points, preserve_range=True)
        e = skimage.util.invert(e) # Prepare the image to be super-imposed on the constant image
        smallest_row = None # The smallest row where a corona pixel is detected using THRESHOLD
        smallest_col = (0,REALLY_BIG) # The smallest column where a corona pixel is detected using THRESHOLD
        greatest_row = (0,0) # The greatest row where a corona pixel is detected using THRESHOLD
        greatest_col = (0,0) # The greatest col where a corona pixel is detected using THRESHOLD

        for x in range(len(f)): # For every row in the constant image
            for y in range(len(f[x])): # For every column in that row
                try:
                    val_a = f[x][y]
                except IndexError:
                    val_a = 1
                try:
                    val_b = e[x + offset_x][y + offset_y]
                except IndexError:
                    val_b = 1
                val = min(val_a, val_b)
                if smallest_row is None and val < THRESHOLD:
                    smallest_row = (x,y)
                if val < THRESHOLD and y < smallest_col[1]:
                    smallest_col = (x,y)
                if val < THRESHOLD:
                    greatest_row = (x,y)
                if val < THRESHOLD and y > greatest_col[1]:
                    greatest_col = (x,y)
                f[x][y] = val # Record the super-imposed value into f
        config = Configuration(smallest_row,smallest_col,greatest_row,greatest_col,f,e,j) # Creates a new configuration
        if configurations is None or config < configurations:
            configurations = config
    return configurations
if __name__ == '__main__':
    NUM_CORES = 6
    orig_a = skimage.io.imread('img5.jpg')
    orig_b = skimage.io.imread('img6.jpg')

    a = skimage.color.rgb2gray(orig_a)  # The static, already aligned image in grayscale
    b = skimage.color.rgb2gray(orig_b)  # The modular image that we are trying to align in grayscale
    c = skimage.util.invert(a)  # Invert the image so that black becomes white, and vice-versa
    # g = skimage.util.invert(b)
    configurations = None  # Stores the best configuration

    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(np.array(orig_a, dtype=np.uint8), None)
    kp2, des2 = orb.detectAndCompute(np.array(orig_b, dtype=np.uint8), None)

    del (orig_a)
    del (orig_b)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)
    kp1 = [(point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id) for point in kp1]
    kp2 = [(point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id)   for point in kp2]

    matches = [(mat.queryIdx, mat.trainIdx) for mat in matches]

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

    stored = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(analyze)(mat,c,b,kp1,kp2) for mat in matches)