# Megamovie
The Megamovie Project is analyzing thousands of photos taken during the total solar eclipse in 2017 by citizens. This repo has some codes that make a first alignement and centering process.

megamovie_jul-2.py is reading the picture data from the database results-20180702-133840.csv in order to find the center of the moon as well as its radius following the procedure in https://docs.google.com/document/d/1l5cj76Md_LVBtDQAK-Jz0grdgN6AcDJ9tpsoUHVQoqA/edit Also a first likely position for regulus is found, the final result is a centers-regulus-july2.csv csv file that contains:
Anonymous-photographer-ID,Image-ID,Image-url,Time,Longitude,Latitude,Exposure,X-center,Y-center,X-radius,Y-radius,X-Regulus,Y-Regulus
If you do not have the pictures downloaded, you should use the commented lines where the URL in the database is used for downloading pictures.
fits_generator.py use the results found with the code above for center and resize all the pictures such as the new moon's radius will be 512 pixels in an image with size 6144x6144 with the moon's center in the image's center. In addition, the images are saved as FITS using a first approximation to Helioprojective Coordinates in Sunpy. These new images are heavy ~290 MB.
