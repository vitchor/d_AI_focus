# EXAMPLE :
# python ward.py boat_1.jpeg boat_2.jpeg 0.25 8

import numpy as np
from scipy.cluster.vq import kmeans2

def ICM(data, N, beta):
    print "Performing ICM segmentation..."


    # Initialise segmentation using kmeans
    print "K-means initialisation..."
    clusters, labels = kmeans2(np.ravel(data), N)

    print "Iterative segmentation..."
    f = data.copy()

    def _minimise_cluster_distance(data, labels, N, beta):
        data_flat = np.ravel(data)
        cluster_means = np.array(
            [np.mean(data_flat[labels == k]) for k in range(N)]
            )
        variance = np.sum((data_flat - cluster_means[labels])**2) \
                   / data_flat.size

        # How many of the 8-connected neighbouring pixels are in the
        # same cluster?
        count = np.zeros(data.shape + (N,), dtype=int)
        count_inside = count[1:-1, 1:-1, :]

        labels_img = labels.reshape(data.shape)
        for k in range(N):
            count_inside[..., k] += (k == labels_img[1:-1:, 2:])
            count_inside[..., k] += (k == labels_img[2:, 1:-1])
            count_inside[..., k] += (k == labels_img[:-2, 1:-1])
            count_inside[..., k] += (k == labels_img[1:-1, :-2])

            count_inside[..., k] += (k == labels_img[:-2, :-2])
            count_inside[..., k] += (k == labels_img[2:, 2:])
            count_inside[..., k] += (k == labels_img[:-2, 2:])
            count_inside[..., k] += (k == labels_img[2:, :-2])

        count = count.reshape((len(labels), N))
        cluster_measure = (data_flat[:, None] - cluster_means)**2 \
                          - beta * variance * count
        labels = np.argmin(cluster_measure, axis=1)

        return cluster_means, labels

    # Initialise segmentation
    cluster_means, labels = _minimise_cluster_distance(f, labels, N, 0)

    stable_counter = 0
    old_label_diff = 0
    i = 0
    while stable_counter < 3:
        i += 1

        cluster_means, labels_ = \
                       _minimise_cluster_distance(f, labels, N, beta)

        new_label_diff = np.sum(labels_ != labels)
        if  new_label_diff != old_label_diff:
            stable_counter = 0
        else:
            stable_counter += 1
        old_label_diff = new_label_diff

        labels = labels_

    print "Clustering converged after %d steps." % i

    return labels.reshape(data.shape)
    

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
# MAIN
###############################################################################
import sys
import time as time
import numpy as np
import scipy as sp
import pylab as pl
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import Ward
from skimage.filter import denoise_tv_chambolle
import scipy.ndimage as ndI
from PIL import Image
#from skimage import filter

#first_image_name = 'boat_1.jpeg'
first_image_name = str(sys.argv[1])

#second_image_name = 'boat_2.jpeg'
second_image_name = str(sys.argv[2])

#scale = 0.25 # from 0 to 1
scale = float(sys.argv[3])

#n_clusters = 8  # number of regions
n_clusters = int(sys.argv[4])

pil_image = Image.open(first_image_name)
width, height = pil_image.size

if scale != 1:
    pil_image = pil_image.resize((int(scale*width), int(scale*height)), Image.ANTIALIAS)

first_image = np.asarray(pil_image)
first_image_final = np.copy(first_image)
first_image = pl.mean(first_image,2)

#first_image_edges = filter.canny(first_image, sigma=5)

X = np.reshape(first_image, (-1, 1))

###############################################################################
# Define the structure A of the data. Pixels connected to their neighbors.
connectivity = grid_to_graph(*first_image.shape)

###############################################################################
# Compute clustering
print("Compute structured hierarchical clustering...")
st = time.time()

# first_label = np.copy(first_image)
# 
# first_label_3d = ICM(first_image_final, 2, 8)
# 
# for label_index_1 in range(len(first_label_3d)):
#     for label_index_2 in range(len(first_label_3d[label_index_1])):
#         
#         row = first_label_3d[label_index_1][label_index_2]
#         first_label[label_index_1][label_index_2] = int(row[0] * 1 + row[1] * 2 + row[2] * 4 + row[3] * 8)
         
    

# sp.misc.imsave('ICM.png', first_label)
# np.set_printoptions(threshold='nan')



ward = Ward(n_clusters=n_clusters, connectivity=connectivity, compute_full_tree=False).fit(X)
first_label = np.reshape(ward.labels_, first_image.shape)



first_image_cluster_deltas = []

for cluster_index in range(n_clusters):
    
    new_image = np.copy(first_image)
    cluster_points = []
    
    for index_1 in range(len(first_label)):
        for index_2 in range(len(first_label[index_1])):
            
            if first_label[index_1][index_2] != cluster_index:
                new_image[index_1][index_2] = -1
            else :
                cluster_points.append(new_image[index_1][index_2])
            
    
    #histogram = np.histogram(cluster_points)
    #splitted_arrays = np.array_split(histogram[0], 4)
    
    #maxValue = max([max(splitted_arrays[0]), max(splitted_arrays[3])])
    #minValue = min([min(splitted_arrays[1]), min(splitted_arrays[2])])
    
    if len(cluster_points) > 0:
        average = sum(cluster_points)/len(cluster_points)
        delta = np.std(cluster_points)# / average
    else:
        delta = 0
    
    first_image_cluster_deltas.append(delta)
    
    #sp.misc.imsave(first_image_name +"_" + str(len(cluster_points)) + "_" + str(cluster_index) + '_autosave.png', new_image)
          
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
#SECOND IMAGE
###############################################################################
# Generate data

pil_image = Image.open(second_image_name)
pil_image = pil_image.resize((int(scale*width), int(scale*height)), Image.ANTIALIAS)

second_image = np.asarray(pil_image)

second_image_final = np.copy(second_image)
second_image = pl.mean(second_image,2)

#second_image_edges = filter.canny(second_image, sigma=4)

#X = np.reshape(second_image, (-1, 1))


X = np.reshape(second_image, (-1, 1))


# Define the structure A of the data. Pixels connected to their neighbors.
connectivity = grid_to_graph(*second_image.shape)

# Compute clustering
print("Compute structured hierarchical clustering...")
st = time.time()



second_label = np.copy(second_image)

#second_label_3d = ICM(second_image_final, 2, 8)

# for label_index_1 in range(len(second_label_3d)):
#     for label_index_2 in range(len(second_label_3d[label_index_1])):
#         
#         row = second_label_3d[label_index_1][label_index_2]
#         second_label[label_index_1][label_index_2] = int(row[0] * 1 + row[1] * 2 + row[2] * 4 + row[3] * 8)

ward = Ward(n_clusters=n_clusters, connectivity=connectivity, compute_full_tree=False).fit(X)
second_label = np.reshape(ward.labels_, second_image.shape)

np.set_printoptions(threshold='nan')

second_image_cluster_deltas = []

for cluster_index in range(n_clusters):

    new_image = np.copy(second_image)
    cluster_points = []

    for index_1 in range(len(second_label)):
        for index_2 in range(len(second_label[index_1])):

            if second_label[index_1][index_2] != cluster_index:
                new_image[index_1][index_2] = -1
            else :
                cluster_points.append(new_image[index_1][index_2])


    #histogram = np.histogram(cluster_points)
    #splitted_arrays = np.array_split(histogram[0], 4)

    #maxValue = max([max(splitted_arrays[0]), max(splitted_arrays[3])])
    #minValue = min([min(splitted_arrays[1]), min(splitted_arrays[2])])
    
    print "aqui" 
    print len(cluster_points)
    #delta = (maxValue - minValue) * (max(histogram[1]) - min(histogram[1]))
    
    if len(cluster_points) > 0:
        average = sum(cluster_points)/len(cluster_points)
        delta = np.std(cluster_points)# / average
    else:
        delta = 0
        
    print delta
    
    second_image_cluster_deltas.append(delta)
    
    #sp.misc.imsave(second_image_name +"_" + str(len(cluster_points)) + "_" + str(cluster_index) + '_autosave.png', new_image)

pl.figure(figsize=(5, 5))
pl.imshow(first_image_final)
for l in range(n_clusters):
   pl.contour(first_label == l, contours=1,
             colors=[pl.cm.spectral(l / float(n_clusters)), ])
pl.xticks(())
pl.yticks(())

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
# FINAL IMAGE
###############################################################################

#print label
print("Elapsed time: ", time.time() - st)
print("Number of pixels: ", second_label.size)
print("Number of clusters: ", np.unique(second_label).size)


print first_image_cluster_deltas
print second_image_cluster_deltas

final_image = np.copy(first_image_final)
test_image = np.copy(first_image)

for index_1 in range(len(final_image)):
    for index_2 in range(len(final_image[index_1])):
        
        first_image_cluster_value = int(first_label[index_1][index_2])
        first_image_delta = first_image_cluster_deltas[first_image_cluster_value]
        
        second_image_cluster_value = int(second_label[index_1][index_2])
        second_image_delta = second_image_cluster_deltas[second_image_cluster_value]
        
        #compare modules
        if first_image_delta > second_image_delta:
            final_image[index_1][index_2] = first_image_final[index_1][index_2]
        else:
            final_image[index_1][index_2] = second_image_final[index_1][index_2]
            
        
        
pl.figure(figsize=(5, 5))
pl.imshow(second_image_final, cmap=pl.cm.gray)
for l in range(n_clusters):
   pl.contour(second_label == l, contours=1,
              colors=[pl.cm.spectral(l / float(n_clusters)), ])
pl.xticks(())
pl.yticks(())

#final_image = ndI.median_filter(final_image, 2)
#from scikits.image.filter import tv_denoise
#from tv_denoise import tv_denoise
#final_image = denoise_tv_chambolle(final_image, weight=0.1, multichannel=True)
sp.misc.imsave("output/" + second_image_name + "_" + str(cluster_index + 1) + '_final.png', final_image)

pl.show()

