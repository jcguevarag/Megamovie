# Import all necessary libraries. Skimage is shorthand for scikit-image
import skimage.io
import skimage.util
import skimage.color
import skimage.transform
import copy
from matplotlib import pyplot as plt
from configuration import Configuration
from datetime import datetime


THRESHOLD = 0.2 # The highest pixel value that will be considered as part of the corona
REALLY_BIG = 10 # Placeholder value. Can be changed to anything above 1.
ROTATE_BY = 1 # Angle rotation step. Change this to change the degrees you want to rotate it by.


a = skimage.color.rgb2gray(skimage.io.imread('5853.png')) # The static, already aligned image in grayscale
b = skimage.color.rgb2gray(skimage.io.imread('5854.png')) # The modular image that we are trying to align in grayscale
c = skimage.util.invert(a) # Invert the image so that black becomes white, and vice-versa
configurations = None # Stores the best configuration
stored = [] # Can save every _ time steps to manually confirm results

for j in range(0,360,ROTATE_BY): # Rotate up to 360 Degrees by the Angle Step
    print('We are on degree: {}'.format(str(j)), 'The Time is: {}'.format(str(datetime.now())))
    f = copy.deepcopy(c)
    e = skimage.transform.rotate(b, j, preserve_range=True)
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
                val_b = e[x][y]
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
    if j % 10 == 0: # Stores every 10 angle steps. Can be changed by changing the number. Can also make a variable.
        stored.append(config)

skimage.io.imshow(configurations.superimposed_image) # Shows the best configuration
plt.show() # Shows the plot
# time.sleep(30)