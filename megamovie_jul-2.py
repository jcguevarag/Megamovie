#!/usr/bin/env python

"""
This code finds the center positions for the moon in the pictures taken for the Megamovie project.
A detailed description of how does the code works can be found in https://docs.google.com/document/d/1l5cj76Md_LVBtDQAK-Jz0grdgN6AcDJ9tpsoUHVQoqA/edit?usp=sharing

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
import cv2
import rawpy
from IPython.display import clear_output
from IPython import display
import time,os,sys
import datetime
from os import listdir
from os.path import isfile, join

#It is set a function to convert from RGB to grayscale according to https://www.mathworks.com/help/matlab/ref/rgb2gray.html
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
#It is set the actual directory as well as the CSV containing the database of the pictures
dir = os.getcwd()
dir_c = os.path.join(dir)
with open('results-20180702-133840.csv', 'r') as f:
	reader = csv.reader(f)
	your_list = list(reader)
	f.close()
#Creating the output file that will store the center positions and radii found
fa = open(dir_c+'/centers-regulus-july2.csv','w')
print('Anonymous-photographer-ID','Image-ID','Image-url','Time','Longitude','Latitude','Exposure','X-center','Y-center','X-radius','Y-radius','X-Regulus','Y-Regulus',sep=',',file=fa)
#Download, and center finding
n_regulus = 0
n_radios = 0
n_imagen = 0
for idx,item in enumerate(your_list[1:]):
	ano_phid = item[0]	
	pho_id = item[6]
	lgtd = item[3]
	lttd = item[17]
	tmpic = item[16]
	store_uri = item[12]
	im_name = pho_id
	exp_time = item[13]
	im_fmt = item[12][-3:]
	n_imagen = n_imagen+1
	print('processing im %i with name %s'%(n_imagen,str(im_name)))
	im_url = item[12]
#	f =  dir_c+'/images/%s.%s'%(im_name,im_fmt)
#	urllib.request.urlretrieve(im_url, f)
	f = '/Volumes/data/Megamovie/eclipse_megamovie/%s.%s'%(im_name,im_fmt)
	#Reading image according to the format
	try:
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
		#Gaussian Blur processing for smoothing boundaries
		im_gray = cv2.GaussianBlur(im_gs,(0,0),3)
		"Horizontal and vertical cutting limb position searching"
		#Vertical cutting
		indices_v=[]
		im_t = np.transpose(im_gray)
		for idx,item in enumerate(im_t):
			grad_im = np.gradient(item)
			std_grad = np.std(grad_im)
			mean_grad = np.mean(grad_im)
			max_grad_im = np.max(grad_im)
			rms_grad = np.sqrt(np.mean(np.square(grad_im)))
			min_grad_im = np.min(grad_im)
			l=np.where(grad_im==np.min(grad_im))[0]
			r=np.where(grad_im==np.max(grad_im))[0]
			if len(l)==1 and len(r)==1:
				if max_grad_im>=(rms_grad+(9*std_grad)) and min_grad_im<(rms_grad-(9*std_grad)):
					indices_v.append([l,r])
				else:
					indices_v.append([0,0])
			else:
				indices_v.append([0,0])
		#Horizontal cutting
		indices_h=[]
		for idx,item in enumerate(im_gray):
			grad_im = np.gradient(item)
			std_grad = np.std(grad_im)
			mean_grad = np.mean(grad_im)
			max_grad_im = np.max(grad_im)
			rms_grad = np.sqrt(np.mean(np.square(grad_im)))
			min_grad_im = np.min(grad_im)
			l=np.where(grad_im==np.min(grad_im))[0]
			r=np.where(grad_im==np.max(grad_im))[0]
			if len(l)==1 and len(r)==1:
				if max_grad_im>=(rms_grad+(9*std_grad)) and min_grad_im<(rms_grad-(9*std_grad)):
					indices_h.append([l,r])
				else:
					indices_h.append([0,0])
			else:
				indices_h.append([0,0])	
		"Finding radii from limb position"
		#finding radius and position in X-axis - Vertical cuts (larger difference between limb position is the diamater of the moon)
		radii_v = []
		cts_v = []
		for idx,item in enumerate(indices_v):
			radii_v.append(item[1]-item[0])
			cts_v.append((item[1]+item[0])/2.)
		v_ctr = np.mean(np.where(radii_v==np.nanmax(radii_v)))
		if np.isnan(v_ctr) == True:
			idx_v_center = 0
		else:
			idx_v_center = int(np.mean(np.where(radii_v==np.nanmax(radii_v))))
		r_y = radii_v[idx_v_center]/2.
		c_y = cts_v[idx_v_center]
		#finding radius and position in x-axis Horizontal cuts
		radii_h = []
		cts_h = []
		for idx,item in enumerate(indices_h):
			radii_h.append(item[1]-item[0])
			cts_h.append((item[1]+item[0])/2.)
		h_ctr = np.mean(np.where(radii_h==np.nanmax(radii_h)))
		if np.isnan(h_ctr) == True:
			idx_h_center = 0
		else:
			idx_h_center = int(np.mean(np.where(radii_h==np.nanmax(radii_h))))
		r_x = radii_h[idx_h_center]/2.
		c_x = cts_h[idx_h_center]
		"Masking and scanning far than moon's boundaries for searching stars - Same method of cutting"
		r_avg = (float(r_x)+float(r_y))/2.
		r_df = np.abs(float(r_x)-float(r_y))
		#vertical cuts for stars
		indices_v2=[]
		for idx,item in enumerate(im_t):
			grad_im = np.gradient(item)
			mean_grad = np.mean(grad_im)
			if int(float(c_x)-float(r_avg*1.1))<idx<int(float(c_x)+float(r_avg*1.1)) and r_avg>50 and r_df<50:
				for idg in range(int(float(c_y)-float(r_avg*1.1)),int(float(c_y)+float(r_avg*1.1))):
					try:
						grad_im[idg] = mean_grad 
					except IndexError:
						r_avg = 1.  
			max_grad_im = np.max(grad_im)
			std_grad = np.std(grad_im)
			rms_grad = np.sqrt(np.mean(np.square(grad_im)))
			min_grad_im = np.min(grad_im)
			l=np.where(grad_im==np.min(grad_im))[0]
			r=np.where(grad_im==np.max(grad_im))[0]
			if len(l)==1 and len(r)==1:
				if max_grad_im>=(rms_grad+(9*std_grad)) and min_grad_im<(rms_grad-(9*std_grad)):
					indices_v2.append([l,r])
				else:
					indices_v2.append([np.nan,np.nan])
			else:
				indices_v2.append([np.nan,np.nan])
		#Horizontal cuts for stars
		indices_h2=[]
		for idx,item in enumerate(im_gray):
			grad_im = np.gradient(item)
			mean_grad = np.mean(grad_im)
			if int(float(c_y)-float(r_avg*1.1))<idx<int(float(c_y)+float(r_avg*1.1)) and r_avg>50 and r_df<50:
				for idg in range(int(float(c_x)-float(r_avg*1.1)),int(float(c_x)+float(r_avg*1.1))):
					try:
						grad_im[idg] = mean_grad 
					except IndexError:
						r_avg = 1.  
			max_grad_im = np.max(grad_im)
			std_grad = np.std(grad_im)
			rms_grad = np.sqrt(np.mean(np.square(grad_im)))
			min_grad_im = np.min(grad_im)
			l=np.where(grad_im==np.min(grad_im))[0]
			r=np.where(grad_im==np.max(grad_im))[0]
			if len(l)==1 and len(r)==1:
				if max_grad_im>=(rms_grad+(9*std_grad)) and min_grad_im<(rms_grad-(9*std_grad)):
					indices_h2.append([l,r])
				else:
					indices_h2.append([np.nan,np.nan])
			else:
				indices_h2.append([np.nan])
		"Finding Regulus from the new cutting"
		#Identifying likely Regulus positions with indices_v2 and indices_h2
		reg_x_y_v = []
		for idy,itemy in enumerate(indices_v2):
			if np.isnan(itemy[0]) == False and itemy[1]-itemy[0]<r_y/10:
				d_s_r = np.sqrt(np.abs(float(c_x)-float(idy))**2 + np.abs(float(c_y)-float(np.mean(itemy)))**2)/r_avg
				if 4.7<d_s_r<5.1:
					reg_x_y_v.append([idy,float(np.mean(itemy))])
				else:
					reg_x_y_v.append([0,0])
		reg_x_y_h = []
		for idx,itemx in enumerate(indices_h2):
			if np.isnan(itemx[0]) == False and itemx[1]-itemx[0]<r_x/10:
				d_s_r = np.sqrt(np.abs(float(c_x)-float(np.mean(itemx)))**2 + np.abs(float(c_y)-float(idx))**2)/r_avg
				if 4.7<d_s_r<5.1:
					reg_x_y_h.append([float(np.mean(itemx)),idx])
				else:
					reg_x_y_h.append([0,0])
		#Define "real regulus position" 
		reg_pos = []
		for itemh in reg_x_y_h:
			for itemv in reg_x_y_v:
				if int(itemh[0]) != 0 or int(itemv[0]) != 0:
					if int(itemh[0])==int(itemv[0]) and int(itemh[1])==int(itemv[1]):
						xy = [int(itemv[0]),int(itemv[1])]
						reg_pos.append(xy)
		#Some statistics
		if len(reg_pos) != 0:
			n_regulus = n_regulus + 1
		else:
			xy = [0,0]
			reg_pos.append(xy)
		regulus_x = reg_pos[0][0]
		regulus_y = reg_pos[0][1]
		if r_df<50 and r_avg>50:
			n_radios = n_radios + 1
			print(str(ano_phid),str(im_name),str(store_uri),str(tmpic),float(lgtd),float(lttd),str(exp_time),float(c_x),float(c_y),float(r_x),float(r_y),float(regulus_x),float(regulus_y),sep=',',file=fa)
		else:
			print(str(ano_phid),str(im_name),str(store_uri),str(tmpic),float(lgtd),float(lttd),str(exp_time),np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,sep=',',file=fa)
	except FileNotFoundError:  
		print("No existe el archivo")
print('The python script has finished with %i centers found (%i total images) and %i regulus position found'%(n_radios,n_imagen,n_regulus))
