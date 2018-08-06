import skimage.io
import skimage.transform
import os

os.makedirs('rotated_images_bin_9')

with open('serialized_rotations11.txt', 'r') as rotation_nums:
    lines =  rotation_nums.readlines()
    for img in lines:
        img_data = img.split(":")
        a = skimage.io.imread('images_random_200/'+ img_data[0] + '.jpg')
        a = skimage.transform.rotate(a,int(img_data[1]))
        skimage.io.imsave('rotated_images_bin_9/' + img_data[0] + '_rotated.jpg',a)
