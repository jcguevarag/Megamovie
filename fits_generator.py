#!/usr/bin/env python

"""
This code uses the center positions found for the moon in the pictures taken for the Megamovie project for resizing to 4096x4096 px with radius 512 px, and save as fits.
Author: Juan Camilo Guevara Gomez
Version 0.0: March 2018
Latest version:  July 2018
"""
import multiprocessing as mp
import csv
import matplotlib
import matplotlib.pyplot as plt
import urllib.request
import numpy as np
from matplotlib.patches import Circle
from PIL import Image
import sunpy
import sunpy.map
import sunpy.coordinates
import cv2
import astropy.wcs
from astropy.coordinates import EarthLocation
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel, Gaussian1DKernel
import rawpy
from IPython.display import clear_output
from IPython import display
import time,os,sys
import datetime
#Set actual directory
dir = os.getcwd()
dir_c = os.path.join(dir)
#It is set a function to convert from RGB to grayscale according to https://www.mathworks.com/help/matlab/ref/rgb2gray.html
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
#Input csv files, original database and new databse with centers
with open('results-20180702-133840.csv', 'r') as f:
	reader = csv.reader(f)
	db_org = list(reader)
	f.close()
with open('centers-regulus-july2.csv', 'r') as f:
	reader = csv.reader(f)
	db_ctrs = list(reader)
	f.close()
#Resizing and centering
im_total = len(db_ctrs[1:])
nimp = 0
for idx,item in enumerate(db_ctrs[2513:]):
	if item[7]!='nan' and item[8]!='nan' and item[9]!='nan' and item[10]!='nan':
		nimp = nimp+1
		print("processing image %i of %i"%(2513+idx,im_total))
		im_name = item[1]
		im_author = item[0]
		im_fmt = item[2][-3:]
		im_time = item[3][:10]+'T'+item[3][11:19]
		im_lon = item[4]
		im_lat = item[5]
		im_exp = item[6]
		f = '/Volumes/data/Megamovie/eclipse_megamovie/%s.%s'%(im_name,im_fmt)
		#read image downloaded
		if im_fmt == 'jpg':
			im_rgb = np.flipud(matplotlib.image.imread(f))
			im_gs = rgb2gray(im_rgb)
			print(im_fmt, 'format')
		elif im_fmt == 'tif':
			im_rgb = np.array(Image.open(f))
			im_gs = rgb2gray(im_rgb)
			print(im_fmt, 'format')
		else:
			raw = rawpy.imread(f)
			rgb_raw = raw.postprocess(no_auto_bright=True,use_auto_wb =False,gamma=None)
			im_gs = rgb2gray(rgb_raw)
			print(im_fmt, 'format')
		h,w = np.shape(im_gs)[:2]
		#read original radii
		rix = eval(item[9])
		riy = eval(item[10])
		#desired Sun's radii for all images
		rfx = 512.
		rfy = 512.
		#scale constant
		kxf = rfx/rix
		kyf = rfy/riy
		#scaling image according to scales constants
		res1 = cv2.resize(im_gs,(int(w*kxf),int(h*kyf)), interpolation = cv2.INTER_CUBIC)
		hm,wm = res1.shape[:2]
		#Define new image size to9000x6000 (New radii will be kept)
		nh = 3072*2
		nw = 3072*2
		#Read original centers in X and Y
		xi = eval(item[7])
		yi = eval(item[8])
		#Centering the Sun's image in New image with size 4096x4096
		kx = nw/wm
		xn = kxf*xi
		ky = nh/hm
		yn = kyf*yi
		xc = 3072
		yc = 3072
		tx = xc - xn
		ty = yc - yn
		#Creating final image with new conditions
		print('Resizing image to 4096x4096')
		M = np.float32([[1,0,tx],[0,1,ty]])
		final = cv2.warpAffine(res1,M,(nw,nh))
		#Creating WCS
		print('Creating sunpy map')
		dsun = sunpy.coordinates.get_sunearth_distance(im_time)
		rsun_obs = np.arctan(sunpy.sun.constants.radius / dsun).to('arcsec')
		im_radius = 512. * u.pix
		plate_scale = rsun_obs / im_radius
		loc = EarthLocation(lat=im_lat, lon=im_lon)
		fudge_angle = 0.0 * u.deg # update this in case your camera was not perfectly level.
		solar_rotation_angle = sunpy.coordinates.get_sun_orientation(loc, im_time) + fudge_angle
		hgln_obs = 0 * u.deg # sunpy.coordinates.get_sun_L0(time)
		hglt_obs = sunpy.coordinates.get_sun_B0(im_time)
		w = astropy.wcs.WCS(naxis=2)
		w.wcs.crpix = [xc, yc]
		w.wcs.cdelt = np.ones(2) * plate_scale.to('arcsec/pix').value
		w.wcs.crval = [0, 0]
		w.wcs.ctype = ['TAN', 'TAN']
		w.wcs.cunit = ['arcsec', 'arcsec']
		w.wcs.dateobs = im_time
		#creating header
		header = dict(w.to_header())
		header.update({'CROTA2': solar_rotation_angle.to('deg').value})
		header.update({'DSUN_OBS': dsun.to('m').value})
		header.update({'HGLN_OBS': hgln_obs.to('deg').value})
		header.update({'HGLT_OBS': hglt_obs.to('deg').value})
		header.update({'CTYPE1': 'HPLN-TAN'})
		header.update({'CTYPE2': 'HPLT-TAN'})
		header.update({'RSUN': dsun.to('m').value})
		header.update({'RSUN_OBS': np.arctan(sunpy.sun.constants.radius / dsun).to('arcsec').value})
		header.update({'AUTHOR': im_author})
		header.update({'EXPTIME': im_exp})
		for ind,elem in enumerate(db_org):
			if elem[6] == im_name:
				header.update({'CAMERA_MODEL': str(elem[1])})
				header.update({'CAMERA_MAKE': str(elem[5])})
		#Creating map and saving FITS
		im_map = sunpy.map.Map(final,header)
		im_map.save(dir_c+'/FITS_level_0/%s_level_0.fits'%im_name,filetype='auto')
print('The program has finished and %i FITS files have been saved on FITS_level_0 folder'%nimp)	
