import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib
from PIL import Image
import glob
import os
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage import exposure


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

# def main():
#     im = Image.open('pelicula_msngf_hdr_movil_window/0.jpg')
#     im = im.convert('RGB')
#     data = np.array(im)
#     polar_grid, r, theta, ccc = reproject_image_into_polar(data)
    # plot_polar_image(data, origin=None)
    #plot_directional_intensity(data, origin=None)

    # plt.show()

def plot_directional_intensity(data, origin=None):
    """Makes a cicular histogram showing average intensity binned by direction
    from "origin" for each band in "data" (a 3D numpy array). "origin" defaults
    to the center of the image."""
    def intensity_rose(theta, band, color):
        theta, band = theta.flatten(), band.flatten()
        intensities, theta_bins = bin_by(band, theta)
        mean_intensity = map(np.mean, intensities)
        width = np.diff(theta_bins)[0]
        plt.bar(theta_bins, mean_intensity, width=width, color=color)
        plt.xlabel(color + ' Band')
        plt.yticks([])

    # Make cartesian coordinates for the pixel indicies
    # (The origin defaults to the center of the image)
    x, y = index_coords(data, origin)

    # Convert the pixel indices into polar coords.
    r, theta = cart2polar(x, y)

    # Unpack bands of the image
    red, green, blue = data.T

    # Plot...
    plt.figure()

    plt.subplot(2,2,1, projection='polar')
    intensity_rose(theta, red, 'Red')

    plt.subplot(2,2,2, projection='polar')
    intensity_rose(theta, green, 'Green')

    plt.subplot(2,1,2, projection='polar')
    intensity_rose(theta, blue, 'Blue')

    plt.suptitle('Average intensity as a function of direction')

def plot_polar_image(data, origin=None):
    """Plots an image reprojected into polar coordinages with the origin
    at "origin" (a tuple of (x0, y0), defaults to the center of the image)"""
    r, theta, polar_grid= reproject_image_into_polar(data, origin)
    theta=np.degrees(theta)+180.
    r = r * (1920/1024.)
    plt.figure()
    plt.imshow(polar_grid, extent=(theta.min(), theta.max(), r.max(), r.min()))
    plt.axis('auto')
    plt.ylim(plt.ylim()[::-1])
    plt.xlabel('Theta Coordinate (degrees)')
    plt.ylabel('R Coordinate (arcsec)')
    plt.title('Image in Polar Coordinates')
    plt.show()

def plot_polar(data,r,theta):
    """Plots an image reprojected into polar coordinages with the origin
    at "origin" (a tuple of (x0, y0), defaults to the center of the image)"""
    # r, theta, polar_grid= reproject_image_into_polar(data, origin)
    # theta=np.degrees(theta)+180.
    # r = r * (1920/1024.)
    plt.figure()
    plt.imshow(data, extent=(theta.min(), theta.max(), r.max(), r.min()))
    plt.axis('auto')
    plt.ylim(plt.ylim()[::-1])
    plt.xlabel('Theta Coordinate (degrees)')
    plt.ylabel('R Coordinate (arcsec)')
    plt.title('Image in Polar Coordinates')
    plt.show()

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


def bin_by(x, y, nbins=30):
    """Bin x by y, given paired observations of x & y.
    Returns the binned "x" values and the left edges of the bins."""
    bins = np.linspace(y.min(), y.max(), nbins+1)
    # To avoid extra bin for the max value
    bins[-1] += 1 

    indicies = np.digitize(y, bins)

    output = []
    for i in range(1, len(bins)):
        output.append(x[indicies==i])

    # Just return the left edges of the bins
    bins = bins[:-1]

    return output, bins

def reproject_image_into_polar(data, origin=None):
    """Reprojects a 3D numpy array ("data") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image."""
    ny, nx = data.shape[:2]
    if origin is None:
        origin = (nx//2, ny//2)

    # Determine that the min and max r and theta coords will be...
    x, y = index_coords(data, origin=origin)
    r, theta = cart2polar(x, y)

    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i = np.linspace(r.min(), r.max(), nx)#*(1920/1024)
    theta_i = np.linspace(theta.min(), theta.max(), ny)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    xi, yi = polar2cart(r_grid, theta_grid)
    xi += origin[0] # We need to shift the origin back to 
    yi += origin[1] # back to the lower-left corner...
    coords2 = np.vstack((xi, yi))
    xi, yi = xi.flatten(), yi.flatten()
    coords = np.vstack((xi, yi)) # (map_coordinates requires a 2xn array)

    # Reproject each band individually and the restack
    # (uses less memory than reprojection the 3-dimensional array in one step)
    zi = sp.ndimage.map_coordinates(data, coords, order=1)
    bands = zi.reshape((nx, ny))
    # bands = []
    # for band in data.T:
    #     zi = sp.ndimage.map_coordinates(band, coords, order=1)
    #     bands.append(zi.reshape((nx, ny)))
    # output = np.dstack(bands)
    return r_i, theta_i,bands

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.11])

if __name__ == '__main__':
    path = 'pelicula_msngf_hdr_movil_window/'
    # files = glob(path+'*jpg')
    images_names = [img for img in os.listdir(path) if img.endswith(".jpg")]
    def getKey(item):
        return int(item[:-4])
    files = sorted(images_names, key=getKey)
    imref = np.array(Image.open(path+'%s'%files[0]))
    imref = np.array(rgb2gray(imref),dtype=np.float64)/imref.max()
    img_adapteq = exposure.equalize_adapthist(imref)
    meanref = np.array(img_adapteq).mean()
    fc = (2999./360.)
    ang_i = int(eval(input('Input initial angle: '))*fc)
    ang_f = int(eval(input('Input Final angle: '))*fc)
    print(ang_i,ang_f)
    del imref
    jmapa = []
    printProgressBar(0, len(files), prefix = 'Progress:', suffix = 'Complete', length = 50)
    for idx,item in enumerate(files):
        im = np.array(Image.open(path+'%s'%item))
        im = np.array(rgb2gray(im),dtype=np.float64)/im.max()
        img_adapteq = exposure.equalize_adapthist(im)
        # im = im.convert('RGB')
        sc = meanref/np.array(img_adapteq).mean()
        data = np.array(img_adapteq) * sc * 255.
        r, theta,bands = reproject_image_into_polar(data)
        # bstacked = np.dstack(bands)
        # polarg = np.array(rgb2gray(bstacked),dtype=np.float64)
        polarg = bands
        pim = polarg[850:2500]#730
        jm1 = pim[:,ang_i:ang_f]
        jmavg = np.nanmean(jm1,axis=1)
        jmapa.append(jmavg)
        del im,data,bands,polarg,pim,jm1,jmavg
        printProgressBar(idx + 1, len(files), prefix = 'Progress:', suffix = 'Complete', length = 50)
    theta=np.degrees(theta)+180.
    r = r * (1920/1024.)
    r_i = r[730:2100]
    theta_i = theta[ang_i:ang_f]
    del theta,r
    jmapa = np.array(jmapa).T
    jmapa12 = []
    jmapa22 = []
    jmapa32 = []
    for idx,item in enumerate(jmapa.T):
        rms = np.sqrt(np.mean(item**2))
        jmapa12.append(item/rms)
        jmapa22.append(item-rms)
        jmapa32.append(item/np.max(item))
    jmapa12 = np.array(jmapa12)
    jmapa22 = np.array(jmapa22)
    jmapa32 = np.array(jmapa32)
    jmapa1 = []
    jmapa2 = []
    jmapa3 = []
    for idx,item in enumerate(jmapa):
        rms = np.sqrt(np.mean(item**2))
        jmapa1.append(item/rms)
        jmapa2.append(item-rms)
        jmapa3.append(item/np.max(item))
    jmapa1 = np.array(jmapa1)
    jmapa2 = np.array(jmapa2)
    jmapa3 = np.array(jmapa3)
    res1 = np.hstack((jmapa1,jmapa12.T))
    res2 = np.hstack((jmapa2,jmapa22.T))
    res3 = np.hstack((jmapa3,jmapa32.T))
    resj = np.hstack((jmapa,jmapa/np.mean(jmapa)))


    # main()