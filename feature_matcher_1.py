import numpy as np
import cv2
from matplotlib import pyplot as plt


img1 = cv2.imread('img5.jpg',0) # queryImage
img2 = cv2.imread('img6.jpg',0) # trainImage


print(img1)
print(img2)
# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=2)

plt.imshow(img3),plt.show()

for mat in matches:
    img1_idx = mat.queryIdx
    img2_idx = mat.trainIdx

    print('Img 1 Point: {}'.format(str([int(x) for x in kp1[img1_idx].pt])))
    print('Img 2 Point: {}'.format(str([int(x) for x in kp2[img2_idx].pt])))


