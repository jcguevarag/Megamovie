from joblib import Parallel, delayed
import multiprocessing as mp
import numpy as np
import random
import cv2
import os
from tkinter import *
import time
import matplotlib
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.util import random_noise
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
from itertools import islice

def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def linearWeight(pixel_value):
    """ Linear weighting function based on pixel intensity that reduces the
    weight of pixel values that are near saturation.

    Parameters
    ----------
    pixel_value : np.uint8
        A pixel intensity value from 0 to 255

    Returns
    -------
    weight : np.float64
        The weight corresponding to the input pixel intensity

    """
    z_min, z_max = 0.,255.
    if pixel_value <= (z_min + z_max) / 2:
        w = pixel_value - z_min
    else:
        w = z_max - pixel_value

    return w
    #     return pixel_value - z_min
    # return z_max - pixel_value


def sampleIntensities(images):
    """Randomly sample pixel intensities from the exposure stack.

    Parameters
    ----------
    images : list<numpy.ndarray>
        A list containing a stack of single-channel (i.e., grayscale)
        layers of an HDR exposure stack

    Returns
    -------
    intensity_values : numpy.array, dtype=np.uint8
        An array containing a uniformly sampled intensity value from each
        exposure layer (shape = num_intensities x num_images)

    """
    z_min, z_max = 0, 255
    num_intensities = z_max - z_min + 1   #values of g(z)
    num_images = len(images)
    intensity_values = np.zeros((num_intensities, num_images), dtype=np.uint8)  #matriz de funcion de respuesta

    # Find the middle image to use as the source for pixel intensity locations
    mid_img = images[num_images // 2]

    for i in range(z_min, z_max + 1):
        rows, cols = np.where(mid_img == i)
        if len(rows) != 0:
            idx = random.randrange(len(rows))
            for j in range(num_images):
                intensity_values[i, j] = images[j][rows[idx], cols[idx]]
    return intensity_values


def computeResponseCurve(intensity_samples, log_exposures, smoothing_lambda, weighting_function):
    """Find the camera response curve for a single color channel

    Parameters
    ----------
    intensity_samples : numpy.ndarray  #Este es lo calculado en funcion Sample Intensities como intensity_values
        Stack of single channel input values (num_samples x num_images)

    log_exposures : numpy.ndarray
        Log exposure times (size == num_images)

    smoothing_lambda : float
        A constant value used to correct for scale differences between
        data and smoothing terms in the constraint matrix -- source
        paper suggests a value of 100.

    weighting_function : callable
        Function that computes a weight from a pixel intensity

    Returns
    -------
    numpy.ndarray, dtype=np.float64
        Return a vector g(z) where the element at index i is the log exposure
        of a pixel with intensity value z = i (e.g., g[0] is the log exposure
        of z=0, g[1] is the log exposure of z=1, etc.)
    """
    z_min, z_max = 0, 255
    intensity_range = 255  # difference between min and max possible pixel value for uint8
    num_samples = intensity_samples.shape[0]
    num_images = len(log_exposures)

    # NxP + [(Zmax-1) - (Zmin + 1)] + 1 constraints; N + 256 columns
    mat_A = np.zeros((num_images * num_samples + intensity_range, num_samples + intensity_range + 1), dtype=np.float64)
    mat_b = np.zeros((mat_A.shape[0], 1), dtype=np.float64)

    # 1. Add data-fitting constraints:
    print('Computing response curve step1')
    k = 0
    for i in range(num_samples):
        for j in range(num_images):
            z_ij = intensity_samples[i, j]
            w_ij = weighting_function(z_ij)
            mat_A[k, z_ij] = w_ij
            mat_A[k, (intensity_range + 1) + i] = -w_ij
            mat_b[k, 0] = w_ij * log_exposures[j]
            k += 1
    print('Computing response curve step2')
    # 2. Add smoothing constraints:
    for z_k in range(z_min + 1, z_max):
        w_k = weighting_function(z_k)
        mat_A[k, z_k - 1] = w_k * smoothing_lambda
        mat_A[k, z_k    ] = -2 * w_k * smoothing_lambda
        mat_A[k, z_k + 1] = w_k * smoothing_lambda
        k += 1

    # 3. Add color curve centering constraint:
    mat_A[k, (z_max - z_min) // 2] = 1
    print('Solving matrix response curve')
    inv_A = np.linalg.pinv(mat_A)
    x = np.dot(inv_A, mat_b)

    g = x[0: intensity_range + 1]
    return g[:, 0]


def computeRadianceMap(images, log_exposure_times, response_curve, weighting_function):
    """Calculate a radiance map for each pixel from the response curve.

    Parameters
    ----------
    images : list
        Collection containing a single color layer (i.e., grayscale)
        from each image in the exposure stack. (size == num_images)

    log_exposure_times : numpy.ndarray
        Array containing the log exposure times for each image in the
        exposure stack (size == num_images)

    response_curve : numpy.ndarray
        Least-squares fitted log exposure of each pixel value z

    weighting_function : callable
        Function that computes the weights

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        The image radiance map (in log space)
    """
    img_shape = images[0].shape
    img_rad_map = np.zeros(img_shape, dtype=np.float64)
    num_images = len(images)
    print('Computing radiance map')
    # printProgressBar(0, img_shape[0], prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            g = np.array([response_curve[images[k][i, j]] for k in range(num_images)])
            w = np.array([weighting_function(images[k][i, j]) for k in range(num_images)])
            SumW = np.sum(w)
            if SumW > 0:
                img_rad_map[i, j] = np.sum(w * (g - log_exposure_times) / SumW)
            else:
                img_rad_map[i, j] = g[num_images // 2] - log_exposure_times[num_images // 2]
            # printProgressBar(i + 1, img_shape[0], prefix = 'Progress:', suffix = 'Complete', length = 50)
    return np.array(img_rad_map,dtype=np.complex)


def r_c_parallel(i,images, log_exposure_times, response_curve, weighting_function):
    """Calculate a radiance map for each pixel from the response curve.

    Parameters
    ----------
    images : list
        Collection containing a single color layer (i.e., grayscale)
        from each image in the exposure stack. (size == num_images)

    log_exposure_times : numpy.ndarray
        Array containing the log exposure times for each image in the
        exposure stack (size == num_images)

    response_curve : numpy.ndarray
        Least-squares fitted log exposure of each pixel value z

    weighting_function : callable
        Function that computes the weights

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        The image radiance map (in log space)
    """
    img_shape = images[0].shape
    img_rad_map = np.zeros(img_shape, dtype=np.float64)
    num_images = len(images)
    print('Making row %i'%i)
    # printProgressBar(0, img_shape[0], prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    # for i in range(img_shape[0]):
    for j in range(img_shape[1]):
        g = np.mean(np.array([response_curve[images[k][i, j]] for k in range(num_images)]),axis=0)
        w = np.mean(np.array([weighting_function(images[k][i, j]) for k in range(num_images)]),axis=0)
        SumW = np.sum(w)
        if SumW > 0:
            img_rad_map[i, j] = np.sum(w * (g - log_exposure_times) / SumW)
            # return img_rad_map
        else:
            img_rad_map[i, j] = g[num_images // 2] - log_exposure_times[num_images // 2]
            # return img_rad_map
            # printProgressBar(i + 1, img_shape[0], prefix = 'Progress:', suffix = 'Complete', length = 50)
    return np.array(img_rad_map,dtype=np.complex)



def globalToneMapping(image, gamma):
    """Global tone mapping using gamma correction
    ----------
    images : <numpy.ndarray>
        Image needed to be corrected
    gamma : floating number
        The number for gamma correction. Higher value for brighter result; lower for darker
    Returns
    -------
    numpy.ndarray
        The resulting image after gamma correction
    """
    # image_corrected = cv2.pow(image/255., 1.0/gamma)
    image_corrected = np.abs(np.power(image/255., 1.0/gamma))
    return image_corrected


def intensityAdjustment(image, template):
    """Tune image intensity based on template
        ----------
        images : <numpy.ndarray>
            image needed to be adjusted
        template : <numpy.ndarray>
            Typically we use the middle image from image stack. We want to match the image
            intensity for each channel to template's
        Returns
        -------
        numpy.ndarray
            The resulting image after intensity adjustment
        """
    m, n = image.shape
    channel =1
    output = np.zeros((m, n, channel))
    for ch in range(channel):
        image_avg, template_avg = np.average(image[:, :, ch]), np.average(template[:, :, ch])
        output[..., ch] = image[..., ch] * (template_avg / image_avg)

    return output


def computeHDR(images, log_exposure_times, smoothing_lambda=100., gamma=1.6):
    """Computational pipeline to produce the HDR images
    ----------
    images : list<numpy.ndarray>
        A list containing an exposure stack of images
    log_exposure_times : numpy.ndarray
        The log exposure times for each image in the exposure stack
    smoothing_lambda : np.int (Optional)
        A constant value to correct for scale differences between
        data and smoothing terms in the constraint matrix -- source
        paper suggests a value of 100.
    Returns
    -------
    numpy.ndarray
        The resulting HDR with intensities scaled to fit uint8 range
    """

    num_channels = 1
    hdr_image = np.zeros(images[0].shape, dtype=np.float64)

    layer_stack = [img[:, :] for img in images]
    intensity_samples = sampleIntensities(layer_stack)
    response_curve = computeResponseCurve(intensity_samples, log_exposure_times, smoothing_lambda, linearWeight)

    # img_shape = layer_stack[0].shape
    # img_rad_map = np.zeros(img_shape, dtype=np.float64)
    # inputs = range(img_shape[0])
    

    # num_cores = multiprocessing.cpu_count()

    # img_rad_map_par = Parallel(n_jobs=num_cores)(delayed(r_c_parallel)(i,layer_stack, log_exposure_times, response_curve, linearWeight) for i in inputs)
    # print(img_rad_map_par)
    # img_rad_map_par = r_c_parallel(inputs,layer_stack, log_exposure_times, response_curve, linearWeight)


    img_rad_map = computeRadianceMap(layer_stack, log_exposure_times, response_curve, linearWeight)
    img_rad_map = img_rad_map*(255./np.max(img_rad_map))
    image_mapped = globalToneMapping(img_rad_map, gamma)
    return image_mapped

    # for channel in range(num_channels):
    #     # Collect the current layer of each input image from the exposure stack
    #     layer_stack = [img[:, :, channel] for img in images]

    #     # Sample image intensities
    #     intensity_samples = sampleIntensities(layer_stack)

    #     # Compute Response Curve
    #     response_curve = computeResponseCurve(intensity_samples, log_exposure_times, smoothing_lambda, linearWeight)

    #     # Build radiance map
    #     img_rad_map = computeRadianceMap(layer_stack, log_exposure_times, response_curve, linearWeight)

    #     # Normalize hdr layer to (0, 255)
    #     hdr_image[..., channel] = cv2.normalize(img_rad_map, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # # Global tone mapping
    # image_mapped = globalToneMapping(hdr_image, gamma)

    # # Adjust image intensity based on the middle image from image stack
    # template = images[len(images)//2]
    # image_tuned = intensityAdjustment(image_mapped, template)

    # # Output image
    # # output = cv2.normalize(image_tuned, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # # output = cv2.normalize(image_mapped, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # return output.astype(np.uint8)

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

def MakeHDR(window,numero):
    images = []
    exp_times = []
    folder1 = 'pelicula_msngf2_hdr_cm1-parallelized/'
    # folder2 = 'pelicula_msngf2_hdr_cm2/'
    print("Making windows number %i"%numero)
    for item in window:
        # print('Reading image %s'%item[-1])
        f = 'pelicula_sun_msngf2/'+item[-1]
        # f = 'pelicula_ima_aug26/'+item[-1]
        im_t = np.flipud(matplotlib.image.imread(f))
        wd,h = im_t.shape[0],im_t.shape[1]
        imgs = np.array(rgb2gray(im_t[int(wd/2)-1500:int(wd/2)+1500,int(h/2)-1500:int(h/2)+1500]),dtype=np.uint8)
        # imgs = np.array(denoise_tv_chambolle(im_gs, weight=0., multichannel=False),dtype=np.uint8)
        images.append(imgs)
        del f,im_t,wd,h,imgs#,im_gs
        exp_times.append(float(item[-2]))
    img_hdr = computeHDR(images,np.log(exp_times))
    img_hdr = np.array((255/np.max(img_hdr))*img_hdr,dtype=np.float64)
    matplotlib.image.imsave(folder1+'%i.jpg'%numero,img_hdr,cmap='gray_r',origin='lower',dpi=900)
    # matplotlib.image.imsave(folder2+'%i.jpg'%idx,img_hdr,cmap='PuOr_r',origin='lower',dpi=900)
    del img_hdr


if __name__ == '__main__':
    metadata = np.load('metadata_aug26.npy')
    n = 20
    windows = []
    for each in window(metadata[1:],n):
        windows.append(list(each))
    pool = mp.Pool(processes=12)
    try:
        jobs = [pool.apply_async(MakeHDR, args=(ventana, numero)) for numero,ventana in enumerate(windows)]
        results = [r.get() for r in jobs]    # This line actually runs the jobs
        pool.close()
        pool.join()
    # re-raise the rest
    except Exception:
        print("Exception in worker:")
        # traceback.print_exc()
        raise
    # metadata_ch = chunkIt(metadata[1:],n)
    # for idx,chun in enumerate(metadata_ch):
    #     images = []
    #     exp_times = []
    #     print('Making chunk %i'%idx)
    #     for item in chun:
    #         print('Reading image %s'%item[-1])
    #         f = 'pelicula_sun_msngf2/'+item[-1]
    #         # f = 'pelicula_ima_aug26/'+item[-1]
    #         im_t = np.flipud(matplotlib.image.imread(f))
    #         wd,h = im_t.shape[0],im_t.shape[1]
    #         imgs = np.array(rgb2gray(im_t[int(wd/2)-1500:int(wd/2)+1500,int(h/2)-1500:int(h/2)+1500]),dtype=np.uint8)
    #         # imgs = np.array(denoise_tv_chambolle(im_gs, weight=0., multichannel=False),dtype=np.uint8)
    #         images.append(imgs)
    #         del f,im_t,wd,h,imgs#,im_gs
    #         exp_times.append(float(item[-2]))
    #     img_hdr = computeHDR(images,np.log(exp_times))
    #     img_hdr = np.array((255/np.max(img_hdr))*img_hdr,dtype=np.float64)
    #     matplotlib.image.imsave(folder1+'%i.jpg'%idx,img_hdr,cmap='gray_r',origin='lower',dpi=900)
    #     matplotlib.image.imsave(folder2+'%i.jpg'%idx,img_hdr,cmap='PuOr_r',origin='lower',dpi=900)
    #     del img_hdr
        # plt.imshow(img_hdr,cmap='viridis',origin='lower')
        # plt.show()
    # n = 21
    # images = []
    # exp_times = []
    # print('Reading images')
    # for item in metadata[1:n]:
    #     print('Reading image %s'%item[-1])
    #     f = 'pelicula_sun_msngf2/'+item[-1]
    #     # f = 'pelicula_ima_aug26/'+item[-1]
    #     im_t = np.flipud(matplotlib.image.imread(f))
    #     wd,h = im_t.shape[0],im_t.shape[1]
    #     imgs = np.array(rgb2gray(im_t[int(wd/2)-1500:int(wd/2)+1500,int(h/2)-1500:int(h/2)+1500]),dtype=np.uint8)
    #     # imgs = np.array(denoise_tv_chambolle(im_gs, weight=0., multichannel=False),dtype=np.uint8)
    #     images.append(imgs)
    #     del f,im_t,wd,h,imgs#,im_gs
    #     exp_times.append(float(item[-2]))
    # img_hdr = computeHDR(images,np.log(exp_times))
    # img_hdr = np.array(img_hdr,dtype=np.float64)
    # matplotlib.image.imsave('HDR_test_denoise_gnf.jpg',img_hdr,dpi=600)
    # plt.imshow(img_hdr,cmap='viridis',origin='lower')
    # plt.show()