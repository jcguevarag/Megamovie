from joblib import Parallel, delayed
import multiprocessing as mp
import csv
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sunpy
import sunpy.map
from scipy import misc
import astropy.wcs
from astropy.coordinates import EarthLocation
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel, Gaussian1DKernel
from scipy import signal

def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def Gaussian_normalization(file,numero):
	st_path = 'Gaussian_normalized_images_JPG_sep4/'
	print("Making image %i of 6001"%numero)
	im_p = np.array(sunpy.map.Map(file).data,dtype=np.float64)
	w = [0.5,1.7,2.5,4,6.5,15,30,85,263,512]
	wd,h = im_p.shape
	k = 0.7
	h = 0.7
	n = len(w)
	gamma = 3.2
	a0=np.nanmin(im_p)
	a1=np.nanmax(im_p)
	Cg=((im_p-a0)/(a1-a0))**(1./gamma)
	C_prima=[]
	for k in w:
		# print("processing w=%f"%k)
		kg2d = Gaussian2DKernel(x_stddev=k)
		BW = im_p - signal.convolve(im_p,kg2d,mode='same',method='fft')
		# BW = im_gs - signal.convolve(im_gs,kg2d,mode='same',method='fft')
		BWW  = signal.convolve(np.square(BW),kg2d,mode='same',method='fft')
		del kg2d
		SW = np.sqrt(BWW)
		del BWW
		C = BW/SW
		del BW,SW
		Cp = np.arctan(k*C)
		C_prima.append(Cp)
		del C,Cp
	del im_p
	Pw=(1-h)*np.nanmean(np.array(C_prima),axis=0)
	I = h*Cg + Pw
	del Pw
	name = file.split('/')[-1].split('.fits')[0]
	misc.imsave(st_path+'%s.jpg'%name,I)
	del I
	# matplotlib.image.imsave(st_path+'%s.jpg'%name,I,cmap='gray',origin='lower',dpi=900)
# print(np.nanmean(I),np.nanmin(I),np.nanmax(I))
# plt.imshow(I,cmap='viridis',origin='lower'),plt.show()



if __name__ == '__main__':
	path = 'FITS_level_0/'
	files = glob.glob(path+'*')
	pool = mp.Pool(processes=2)
	try:
		jobs = [pool.apply_async(Gaussian_normalization, args=(file, numero)) for numero,file in enumerate(files)]
		results = [r.get() for r in jobs]    # This line actually runs the jobs
		pool.close()
		pool.join()
	# re-raise the rest
	except Exception:
		print("Exception in worker:")
		# traceback.print_exc()
		raise