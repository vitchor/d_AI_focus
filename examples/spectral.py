print(__doc__)

# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>, Brian Cheung
# License: BSD 3 clause

import time

import numpy as np
import scipy as sp
import pylab as pl

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering

#lena = sp.misc.lena()
# Downsample the image by a factor of 4

#lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]

lena = pl.imread('best_1.png')
lena = pl.mean(lena,2)
#lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]
lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]

# Convert the image into a graph with the value of the gradient on the
# edges.
graph = image.img_to_graph(lena)

# Take a decreasing function of the gradient: an exponential
# The smaller beta is, the more independent the segmentation is of the
# actual image. For beta=1, the segmentation is close to a voronoi
beta = 5
eps = 1e-6
graph.data = np.exp(-beta * graph.data / lena.std()) + eps

# Apply spectral clustering (this step goes much faster if you have pyamg
# installed)
N_REGIONS = 11

###############################################################################
# Visualize the resulting regions

# 'discretize'
t0 = time.time()
labels = spectral_clustering(graph, n_clusters=N_REGIONS,
                             assign_labels='kmeans',
                             random_state=1)
t1 = time.time()
labels = labels.reshape(lena.shape)

pl.figure(figsize=(5, 5))
pl.imshow(lena,   cmap=pl.cm.gray)
for l in range(N_REGIONS):
    pl.contour(labels == l, contours=1,
               colors=[pl.cm.spectral(l / float(N_REGIONS)), ])
pl.xticks(())
pl.yticks(())
pl.title('Spectral clustering: %s, %.2fs' % ('kmeans', (t1 - t0)))

pl.show()