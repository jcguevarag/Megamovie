#!/usr/bin/env python
# -*- coding: utf-8 -*-

from astropy.io import fits
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.animation as animation
import scipy as sp
import scipy.ndimage
from PIL import Image
from datetime import datetime
import matplotlib.dates as mdates
import pandas as pd
import cv2 

def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

n = 20
windows = []
for each in window(metadata[1:],n):
    windows.append(list(each))
tiempos = []
for idx,item in enumerate(windows):
	tip = [datetime.strptime(it[-3],'%Y-%m-%dT%H:%M:%S.%f') for it in item]
	tiempos.append(easyAverage(tip))

image_folder_ecp = 'pelicula_msngf_hdr_movil_window/'

secuence_C2_data=[]
for i in range(0,20):
	im0_f = os.path.join(image_folder_ecp, '%i.jpg'%i)
	im0_rgb = np.flipud(matplotlib.image.imread(im0_f))
	im0_gs = np.array(rgb2gray(im0_rgb),dtype=np.float64)
	hdulist = fits.open('Diff_C2_SOHO_'+str(i)+'.fts')
	# header=hdulist[0].header
	center_x=1500.  #round(float(header['CRPIX1'])/2) #Solar center in pixels x
	center_y=1500.  #round(float(header['CRPIX2'])/2) #Solar center in pixels y
	scidata = hdulist
	#scidata = scipy.ndimage.gaussian_filter(scidata, sigma=5, order=0)
	lenght=len(hdulist)
	delta_arc_x=1920/1024. #header['CDELT1'] #arc/pixel x
	delta_arc_y=1920/1024. #header['CDELT2'] #arc/pixel y
	rsun=1920/2.#header['RSUN']/2 #Solar radii in arc
	x1=-((center_x*delta_arc_x)/rsun)
	x2=(((lenght-center_x)*delta_arc_x)/rsun)
	y1=-((center_y*delta_arc_y)/rsun)
	y2=(((lenght-center_y)*delta_arc_y)/rsun)
	extent_rad=[x1,x2,y1,y2]
	time = mdates.date2num(tiempos[0])
	# time = mdates.date2num(datetime.strptime(header['DATE-OBS'], "%Y-%m-%dT%H:%M:%S.%f"))
	DATE = '2017-08-21'
	secuence_C2_data.append([scidata,[x1,x2,y1,y2],header['DATE-OBS'],time])

ind=range(0,107)
time_array=np.array(secuence_C2_data)[:,[3]]
time_data = pd.DataFrame({'Time': ind}, index=time_array)
order_time=time_data.sort_index()
order_indexing=order_time.values[:,0]

secuence_C2_data_ordered=[]
for i in order_indexing:
	secuence_C2_data_ordered.append(secuence_C2_data[i])
 
fig = plt.figure()

secuence_C2=[]
for i in range(0,107):
	scidata=secuence_C2_data_ordered[i][0]
	extent_rad=secuence_C2_data_ordered[i][1]
	secuence_C2.append((plt.imshow(scidata, origin='lower',extent=extent_rad, cmap=plt.cm.gray),
	plt.text(-6.0,6.0,'SOHO/LASCO C2: '+ str(secuence_C2_data_ordered[i][2])),))
	plt.xlabel('X [Rsun]')
	plt.ylabel('Y [Rsun]')
	
im_ani = animation.ArtistAnimation(fig, secuence_C2, interval=500, repeat_delay=1000, blit=True)
im_ani.save('animation.gif', writer='imagemagick', fps=5)
plt.show()
 
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def index_coords(data, origin=None):
    """Creates x & y coords for the indicies in a numpy array "data".
    "origin" defaults to the center of the image. Specify origin=(0,0)
    to set the origin to the lower left corner of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_x, origin_y = origin
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x -= origin_x
    y -= origin_y
    return x, y

def cart2polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def polar2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def bin_by(x, y, nbins=1000):
    """Bin x by y, given paired observations of x & y.
    Returns the binned "x" values and the left edges of the bins."""
    bins = np.linspace(y.min(), y.max(), nbins+1)
    # To avoid extra bin for the max value
    bins[-1] += 1 

    indicies = np.digitize(y, bins)

    output = []
    for i in xrange(1, len(bins)):
        output.append(x[indicies==i])

    # Just return the left edges of the bins
    bins = bins[:-1]

    return output, bins

def reproject_image_into_polar(data, origin=(254,256)):
    """Reprojects a 3D numpy array ("data") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin = (nx//2, ny//2)

    # Determine that the min and max r and theta coords will be...
    x, y = index_coords(data, origin=origin)
    r, theta = cart2polar(x, y)

    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i = np.linspace(r.min(), r.max(), nx)
    theta_i = np.linspace(theta.min(), theta.max(), ny)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    xi, yi = polar2cart(r_grid, theta_grid)
    xi += origin[0] # We need to shift the origin back to 
    yi += origin[1] # back to the lower-left corner...
    xi, yi = xi.flatten(), yi.flatten()
    coords = np.vstack((xi, yi)) # (map_coordinates requires a 2xn array)

    # Reproject each band individually and the restack
    # (uses less memory than reprojection the 3-dimensional array in one step)
    band = data
    zi = sp.ndimage.map_coordinates(band, coords, order=1)
    output = zi.reshape((nx, ny))
	
    return output, r_i, theta_i

def plot_polar_image(data, origin=(254,256)):
    """Plots an image reprojected into polar coordinages with the origin
    at "origin" (a tuple of (x0, y0), defaults to the center of the image)"""
    polar_grid, r, theta = reproject_image_into_polar(data, origin)
    plt.imshow(polar_grid, extent=((theta.min()*180)/np.pi, (theta.max()*180)/np.pi, r.max(), r.min()), cmap=plt.cm.gray)
    plt.axis('auto')
    plt.ylim(plt.ylim()[::-1])
    plt.xlabel('Theta Coordinate (radians)')
    plt.ylabel('R Coordinate (pixels)')
    plt.title('Image in Polar Coordinates')
    plt.show()

def jmap_elements(data,CA,n):
	oculter_r=2.0
	total_r=np.mean(np.array(np.abs(data[1])))
	deproject_data,r,theta=reproject_image_into_polar(data[0], (254,256))
	equ_deprojected = cv2.equalizeHist(deproject_data)
	oculter_pix=85
	max_r=round((255*512)/r.max())
	min_r=round((oculter_pix*512)/r.max())
	deproject_data1=equ_deprojected[min_r:max_r]
	t_rad=(CA*np.pi)/180.
	array_rad=np.linspace(theta.min(),theta.max(),512)
	pos=find_nearest(array_rad,t_rad)
	pos_array=[pos-n,pos+n]
	t_min=((array_rad[pos_array[0]])*180)/np.pi
	t_max=((array_rad[pos_array[1]])*180)/np.pi
	partial=np.transpose(deproject_data1)
	partial2=partial[pos_array[0]:pos_array[1]]
	final_deproject=np.transpose(partial2)

	return final_deproject, total_r, oculter_r, t_min, t_max

def jmap_complete_2(CA,delta):
	time=secuence_C2_data[0][2]
	data,max_r,min_r,t_min,t_max=jmap_elements(secuence_C2_data_ordered[0],CA,delta)
	total_data=np.array(data)
	total_Time=[datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f")]
	for i in range(106):
		print(i)
		time=secuence_C2_data[i+1][2]
		data,max_r,min_r,t_min,t_max=jmap_elements(secuence_C2_data_ordered[i+1],CA,delta)
		total_data=np.concatenate((total_data,np.array(data)), axis=1)
		total_Time.append(datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f"))
	
	#~ data=np.transpose(total_data)
	distance=np.linspace(max_r,min_r,np.shape(data)[0])
	
	time=[]
	for i in total_Time:
		time.append(mdates.date2num(i))
	return time, distance, total_data

def jmap_complete(CA,delta):
	total_data=[]
	total_Time=[]
	for k in range(90):
		print(k)
		time=secuence_C2_data_ordered[k][2]
		data,max_r,min_r,t_min,t_max=jmap_elements(secuence_C2_data_ordered[k],CA,delta)
		par_data=np.mean(np.array(data),axis=1)
		total_data.append(par_data)
		total_Time.append(datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f"))	
	data=np.transpose(total_data)
	distance=np.linspace(max_r,min_r,np.shape(data)[0])
	
	time=[]
	for i in total_Time:
		time.append(mdates.date2num(i))
	return time, distance, data

def angular_position_image(data,CA):
	#Over the unprojected image
	
	fig, ax = plt.subplots()
	r=np.linspace(0,x2,100)
	x=[]
	y=[]
	for i in CA:
		x.append(r*np.sin(i*np.pi/180.))
		y.append(r*np.cos(i*np.pi/180.))
	data_test=np.array(data[0])
	equ = cv2.equalizeHist(data_test)
	
	ax.set_xlabel('X [Rsun]')
	ax.set_ylabel('Y [Rsun]')
	ax.set_title('SOHO/LASCO C2:'+ str(data[2]))
	circle=plt.Circle((0, 0), 2.0, color='gray')
	ax.add_artist(circle)
	img=ax.imshow(equ, origin='lower',extent=data[1], cmap=plt.cm.gray)
	for j in range(len(CA)):
		ax.plot(x[j],y[j], color='r')
			
	
	#Over the projected image
	output_data, r, theta=reproject_image_into_polar(data[0])
	oculter_r=2.0
	total_r=np.mean(np.array(np.abs(data[1])))
	oculter_pix=85
	max_r=round((255*512)/r.max())
	min_r=round((oculter_pix*512)/r.max())
	
	deproject=output_data[min_r:max_r]
	equ_d = cv2.equalizeHist(deproject)
	
	y_p=np.linspace(oculter_r,total_r,100)
	x_p=[]
	for i in CA:
		x_p.append(np.ones(100)*i)
	
	fig1 = plt.figure()
	fig1=plt.imshow(equ_d,extent=((theta.min()*180)/np.pi, (theta.max()*180)/np.pi, total_r, oculter_r),cmap=plt.cm.gray)
	for i in range(len(CA)):
		plt.plot(x_p[i],y_p, color='r')
	plt.axis('auto')
	plt.ylim(plt.ylim()[::-1])
	plt.xlabel('Theta Coordinate (degrees)')
	plt.ylabel('R Coordinate (Solar Radius)')
	plt.title('SOHO/LASCO C2:'+ str(data[2])+'\n Image in Polar Coordinates')
	
#>>>>>>>>>>>>>>
	plt.show()

#Main

#Secuencia de interacion
#Plot
secu_interaction=[]
for i in range(19,27):
	data_test=np.array(secuence_C2_data_ordered[i])
	equ = cv2.equalizeHist(data_test[0])
	secu_interaction.append([equ,data_test[1],data_test[2]])
	
#~ front_radius=[0.0,2.29,2.62,3.19,4.26,5.7,7.1,0.0]
front_radius=[0.0,0.0,0.,0.0,2.6,3.0,0.0,0.0]

phi=np.linspace(0,2*np.pi,100)
	
fig, ax = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True,)#figsize=(15,5))
for i in range(4):
	x=front_radius[i]*np.sin(phi)
	y=front_radius[i]*np.cos(phi)
	ax[0][i].plot(x,y,color='b', label=str(front_radius[i])+' Rsun')
	img=ax[0][i].imshow(secu_interaction[i][0], origin='lower',extent=secu_interaction[i][1], cmap=plt.cm.gray)
	ax[0][i].set_title('SOHO/LASCO/C2 - \n'+str(secu_interaction[i][2]))
	circle=plt.Circle((0, 0), 2.0, color='gray')
	#~ ax[0][i].legend()
	ax[0][i].add_artist(circle)
	ax[0][0].set_ylabel('Y [Rsun]')

for i in range(4):
	x=front_radius[i+4]*np.sin(phi)
	y=front_radius[i+4]*np.cos(phi)
	ax[1][i].plot(x,y,color='b',label=str(front_radius[i+4])+' Rsun')
	img=ax[1][i].imshow(secu_interaction[i+4][0], origin='lower',extent=secu_interaction[i+4][1], cmap=plt.cm.gray)
	ax[1][i].set_title('SOHO/LASCO/C2 - \n'+str(secu_interaction[i+4][2]))
	ax[1][0].set_ylabel('Y [Rsun]')
	#~ ax[1][i].legend()
	circle=plt.Circle((0, 0), 2.0, color='gray')
	ax[1][i].add_artist(circle)
	ax[1][i].set_xlabel('X [Rsun]')

#>>>>>>>>>>>>>>
plt.show()

#~ 
#~ #Jmap complete_analisis de altura    
#~ 
ca=[60,70,80,90,100,110,120,130]   
angular_position_image(secuence_C2_data_ordered[22],ca)

jmap=jmap_complete_2(130,7)
data=jmap[2]
max_data=np.max(data)
phi=2.
theta=1.
output_norm=[]
for i in range(len(data)):
	partial=[]
	for j in range(len(data[0])):
		partial.append((max_data/phi)*(data[i][j]/(max_data/theta))**1.0)
	output_norm.append(partial)
data_test=np.array(output_norm)

shape=np.shape(data_test)
r_maxim=jmap[1][-1]
r_minim=jmap[1][0]

plt.imshow(data_test, extent=(0, shape[1], r_maxim, r_minim) ,cmap=plt.cm.gray,  origin='lower')#, vmin=30,vmax=90)
plt.colorbar()
plt.axis('auto')

#>>>>>>>>>>>>>>
plt.show()



#~ 
#~ # High-time plot per angle
#~ data_30=[[24,25,26,27,28,29],[2.8,3.5,4.1,4.7,5.5,6.0]]
#~ data_50_1=[[10,11,12,13,14,15,16,17],[4.2,4.4,4.7,5.0,5.3,5.6,5.7,5.9]]
#~ data_50_2=[[22,23,24,25,26,27],[2.3,2.8,3.5,4.3,5.2,6.0]]
#~ data_70=[[20,21,22,23,24,25,26],[2.3,2.5,2.8,3.3,4.3,5.5,6.2]]
#~ data_90=[[21,22,23,24,25],[2.6,3.1,4.0,5.1,6.2]]
#~ data_110=[[21,22,23,24,25],[2.3,3.1,4.4,5.6,6.3]]
#~ data_130=[[22,23,24,25],[2.7,3.8,4.9,6.0]]
#~ 
#~ def HT(data):
	#~ l=len(data[0])
	#~ time=[]
	#~ for i in range(l):
		#~ t=datetime.strptime(secuence_C2_data_ordered[data[0][i]][2], "%Y-%m-%dT%H:%M:%S.%f")
		#~ time.append(mdates.date2num(t))
	#~ return [time,data[1]]
#~ 
#~ data_30_HT=HT(data_30)
#~ data_50_1_HT=HT(data_50_1)
#~ data_50_2_HT=HT(data_50_2)
#~ data_70_HT=HT(data_70)
#~ data_90_HT=HT(data_90)
#~ data_110_HT=HT(data_110)
#~ data_130_HT=HT(data_130)
#~ 
#~ yerror=0.1 #en radios solares definido como 2 pixeles de error en la determinacion del frente 
#~ time_test1=mdates.date2num(datetime(2010,8,18,0,0,0))
#~ time_test2=mdates.date2num(datetime(2010,8,18,0,0,1))
#~ delta_t=time_test2-time_test1
#~ xerror=delta_t
#~ 
#~ radio_burst_times=[[datetime(2010,8,18,6,0,0),datetime(2010,8,18,6,13,0)],
#~ [datetime(2010,8,18,6,42,0),datetime(2010,8,18,6,54,0)],[datetime(2010,8,18,7,0,0),datetime(2010,8,18,7,19,0)]]
#~ radio_burst_times_index=[]
#~ for i in radio_burst_times:
	#~ partial1=mdates.date2num(i[0])
	#~ partial2=mdates.date2num(i[1])
	#~ radio_burst_times_index.append([partial1,partial2])
#~ 
#~ #Ajustes lineales
#~ 
#~ linear_fit_30=np.polyfit(data_30_HT[0], data_30_HT[1], 1)
#~ linear_fit_50_1=np.polyfit(data_50_1_HT[0], data_50_1_HT[1], 1)
#~ linear_fit_50_2=np.polyfit(data_50_2_HT[0], data_50_2_HT[1], 1)
#~ linear_fit_70=np.polyfit(data_70_HT[0], data_70_HT[1], 1)
#~ linear_fit_90=np.polyfit(data_90_HT[0], data_90_HT[1], 1)
#~ linear_fit_110=np.polyfit(data_110_HT[0], data_110_HT[1], 1)
#~ linear_fit_130=np.polyfit(data_130_HT[0], data_130_HT[1], 1)
#~ 
#~ scale_kms=700000*(1.1574e-5)
#~ vel_fit_30=round(scale_kms*linear_fit_30[0])
#~ vel_fit_50_1=round(scale_kms*linear_fit_50_1[0])
#~ vel_fit_50_2=round(scale_kms*linear_fit_50_2[0])
#~ vel_fit_70=round(scale_kms*linear_fit_70[0])
#~ vel_fit_90=round(scale_kms*linear_fit_90[0])
#~ vel_fit_110=round(scale_kms*linear_fit_110[0])
#~ vel_fit_130=round(scale_kms*linear_fit_130[0])
#~ 
#~ print 'Linear fit speeds'
#~ print vel_fit_30
#~ print vel_fit_50_2
#~ print vel_fit_70
#~ print vel_fit_90
#~ print vel_fit_110
#~ print vel_fit_130
#~ 
#~ colors=['m','c','orange']
#~ 
#~ fig, ax = plt.subplots()
#~ for i in range(len(radio_burst_times_index)):
	#~ ax.axvspan(radio_burst_times_index[i][0],radio_burst_times_index[i][1], facecolor=colors[i], alpha=0.3)
#~ ax.errorbar(data_30_HT[0],data_30_HT[1],xerr=xerror,yerr=yerror, label='Elongation = 30$^\circ$ \n Linear Fit:'+str(vel_fit_30)+' km/s', fmt='--o') 
#ax.errorbar(data_50_1_HT[0],data_50_1_HT[1],xerr=xerror,yerr=yerror, label='Elongation = 50 - CME1',fmt='--o') 
#~ ax.errorbar(data_50_2_HT[0],data_50_2_HT[1],xerr=xerror,yerr=yerror, label='Elongation = 50$^\circ$ \n Linear Fit:'+str(vel_fit_50_2)+' km/s',fmt='--o')
#~ ax.errorbar(data_70_HT[0],data_70_HT[1],xerr=xerror,yerr=yerror, label='Elongation = 70$^\circ$ \n Linear Fit:'+str(vel_fit_70)+' km/s',fmt='--o') 
#~ ax.errorbar(data_90_HT[0],data_90_HT[1],xerr=xerror,yerr=yerror, label='Elongation = 90$^\circ$ \n Linear Fit:'+str(vel_fit_90)+' km/s',fmt='--o') 
#~ ax.errorbar(data_110_HT[0],data_110_HT[1],xerr=xerror,yerr=yerror, label='Elongation = 110$^\circ$ \n Linear Fit:'+str(vel_fit_110)+' km/s',fmt='--o') 
#~ ax.errorbar(data_130_HT[0],data_130_HT[1],xerr=xerror,yerr=yerror, label='Elongation = 130$^\circ$ \n Linear Fit:'+str(vel_fit_130)+' km/s',fmt='--o') 
#~ fig.autofmt_xdate()
#~ ax.xaxis_date()
#~ ax.set_ylabel('Height [Rsun]')
#~ ax.set_xlabel('Time - 2010-08-18 UT')
#~ box = ax.get_position()
#~ ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#~ 
#~ ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#~ plt.show()

# Jmap_bandas definidas por la mediana para varios CA

#~ CA_array1=[60,65,70,75,80,85,90,95]
CA_array1=[60,70,80,90,100,110,120,130]
#~ 
#~ # Para un ancho de 10 grados delta=7
	#~ 
X=[]
Y=[]
data=[]
for i in range(8):
	jmap=jmap_complete(CA_array1[i],7)	
	x,y=np.meshgrid(jmap[0],jmap[1])
	X.append(x)
	Y.append(y)
	data.append(jmap[2])
	Time_sample=jmap[0]

radio_burst_times=[[datetime(2010,8,18,6,0,0),datetime(2010,8,18,6,13,0)],
[datetime(2010,8,18,6,42,0),datetime(2010,8,18,6,54,0)],[datetime(2010,8,18,7,0,0),datetime(2010,8,18,7,19,0)],
[datetime(2010,8,18,7,23,0),datetime(2010,8,18,7,40,0)]]
radio_burst_times_index=[]
for i in radio_burst_times:
	partial1=find_nearest(np.array(Time_sample),mdates.date2num(i[0]))
	partial2=find_nearest(np.array(Time_sample),mdates.date2num(i[1]))
	radio_burst_times_index.append([partial1,partial2])

colors=['m','c','orange','g']

fig, ax = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True,figsize=(25,5))

im0=ax[0][0].pcolormesh(X[0], Y[0], data[0][::-1], cmap=plt.cm.gray, shading='gouraud')
#~ for i in range(len(radio_burst_times_index)):
	#~ ax[0][0].axvspan(Time_sample[radio_burst_times_index[i][0]],Time_sample[radio_burst_times_index[i][1]], facecolor=colors[i], alpha=0.3)
ax[0][0].text(Time_sample[10], 6.0,'SOHO/LASCO/C2 - Elongation: '+str(CA_array1[0]), color='w')
fig.autofmt_xdate()
ax[0][0].xaxis_date()
ax[0][0].set_ylabel('Height [Rsun]')
ax[0][0].axis([X[0].min(), X[0].max(), Y[0].min(), Y[0].max()])

im1=ax[0][1].pcolormesh(X[1], Y[1], data[1][::-1], cmap=plt.cm.gray, shading='gouraud')
#~ for i in range(len(radio_burst_times_index)):
	#~ ax[0][1].axvspan(Time_sample[radio_burst_times_index[i][0]],Time_sample[radio_burst_times_index[i][1]], facecolor=colors[i], alpha=0.3)
ax[0][1].text(Time_sample[10], 6.0,'SOHO/LASCO/C2 - Elongation: '+str(CA_array1[1]), color='w')
fig.autofmt_xdate()
ax[0][1].xaxis_date()
ax[0][1].axis([X[1].min(), X[1].max(), Y[1].min(), Y[1].max()])

im2=ax[0][2].pcolormesh(X[2], Y[2], data[2][::-1], cmap=plt.cm.gray, shading='gouraud')
#~ for i in range(len(radio_burst_times_index)):
	#~ ax[0][2].axvspan(Time_sample[radio_burst_times_index[i][0]],Time_sample[radio_burst_times_index[i][1]], facecolor=colors[i], alpha=0.3)
ax[0][2].text(Time_sample[10], 6.0,'SOHO/LASCO/C2 - Elongation: '+str(CA_array1[2]), color='w')
fig.autofmt_xdate()
ax[0][2].xaxis_date()
ax[0][2].axis([X[2].min(), X[2].max(), Y[2].min(), Y[2].max()])

im3=ax[0][3].pcolormesh(X[3], Y[3], data[3][::-1], cmap=plt.cm.gray, shading='gouraud')
#~ for i in range(len(radio_burst_times_index)):
	#~ ax[0][3].axvspan(Time_sample[radio_burst_times_index[i][0]],Time_sample[radio_burst_times_index[i][1]], facecolor=colors[i], alpha=0.3)
ax[0][3].text(Time_sample[10], 6.0,'SOHO/LASCO/C2 - Elongation: '+str(CA_array1[3]), color='w')
fig.autofmt_xdate()
ax[0][3].xaxis_date()
ax[0][3].axis([X[3].min(), X[3].max(), Y[3].min(), Y[3].max()])

im4=ax[1][0].pcolormesh(X[4], Y[4], data[4][::-1], cmap=plt.cm.gray, shading='gouraud')
#~ for i in range(len(radio_burst_times_index)):
	#~ ax[1][0].axvspan(Time_sample[radio_burst_times_index[i][0]],Time_sample[radio_burst_times_index[i][1]], facecolor=colors[i], alpha=0.3)
ax[1][0].text(Time_sample[10], 6.0,'SOHO/LASCO/C2 - Elongation: '+str(CA_array1[4]), color='w')
fig.autofmt_xdate()
ax[1][0].xaxis_date()
ax[1][0].set_ylabel('Height [Rsun]')
ax[1][0].set_xlabel('Time: 2010-08-18')
ax[1][0].axis([X[4].min(), X[4].max(), Y[4].min(), Y[4].max()])

im5=ax[1][1].pcolormesh(X[5], Y[5], data[5][::-1], cmap=plt.cm.gray, shading='gouraud')
#~ for i in range(len(radio_burst_times_index)):
	#~ ax[1][1].axvspan(Time_sample[radio_burst_times_index[i][0]],Time_sample[radio_burst_times_index[i][1]], facecolor=colors[i], alpha=0.3)
ax[1][1].text(Time_sample[10], 6.0,'SOHO/LASCO/C2 - Elongation: '+str(CA_array1[5]), color='w')
fig.autofmt_xdate()
ax[1][1].xaxis_date()
ax[1][1].set_xlabel('Time: 2010-08-18')
ax[1][1].axis([X[5].min(), X[5].max(), Y[5].min(), Y[5].max()])

im6=ax[1][2].pcolormesh(X[6], Y[6], data[6][::-1], cmap=plt.cm.gray, shading='gouraud')
#~ for i in range(len(radio_burst_times_index)):
	#~ ax[1][2].axvspan(Time_sample[radio_burst_times_index[i][0]],Time_sample[radio_burst_times_index[i][1]], facecolor=colors[i], alpha=0.3)
ax[1][2].text(Time_sample[10], 6.0,'SOHO/LASCO/C2 - Elongation: '+str(CA_array1[6]), color='w')
fig.autofmt_xdate()
ax[1][2].xaxis_date()
ax[1][2].set_xlabel('Time: 2010-08-18')
ax[1][2].axis([X[6].min(), X[6].max(), Y[6].min(), Y[6].max()])

im7=ax[1][3].pcolormesh(X[7], Y[7], data[7][::-1], cmap=plt.cm.gray, shading='gouraud')
#~ for i in range(len(radio_burst_times_index)):
	#~ ax[1][3].axvspan(Time_sample[radio_burst_times_index[i][0]],Time_sample[radio_burst_times_index[i][1]], facecolor=colors[i], alpha=0.3)
ax[1][3].text(Time_sample[10], 6.0,'SOHO/LASCO/C2 - Elongation: '+str(CA_array1[7]), color='w')
fig.autofmt_xdate()
ax[1][3].xaxis_date()
ax[1][3].axis([X[7].min(), X[7].max(), Y[7].min(), Y[7].max()])
ax[1][3].set_xlabel('Time: 2010-08-18')

plt.savefig('jmaps_original.pdf',format='pdf')
#plt.show()
