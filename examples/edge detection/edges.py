import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import pylab as pl
import scipy as sp

from PIL import Image
from skimage import filter

image_name = 'locker_1.png'
pil_image = Image.open(image_name)

pil_image = pl.mean(pil_image,2)

im = np.asarray(pil_image)

edges2 = filter.canny(im, sigma=4)

sp.misc.imsave(image_name + 'edges_auto.png', edges2)