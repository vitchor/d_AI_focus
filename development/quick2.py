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
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries
import matplotlib.cm as cm
#from skimage import filter

def create_segmented_cluster_std_estimate(first_std_matrix, second_std_matrix, transitional_matrix):
    first_image_cluster_segments_std = {}
    first_image_cluster_ocurrence = {}
    
    for index_1 in range(len(first_std_matrix)):
        for index_2 in range(len(first_std_matrix[index_1])):
            deviation_value = first_std_matrix[index_1][index_2]
            
            cluster_tuple = transitional_matrix[index_1][index_2]

            if cluster_tuple in first_image_cluster_segments_std:
                first_image_cluster_segments_std[cluster_tuple] = first_image_cluster_segments_std[cluster_tuple] + deviation_value
                first_image_cluster_ocurrence[cluster_tuple] = first_image_cluster_ocurrence[cluster_tuple] + 1
            else:
                first_image_cluster_segments_std[cluster_tuple] = deviation_value
                first_image_cluster_ocurrence[cluster_tuple] = 1

    for key in first_image_cluster_segments_std:
        first_image_cluster_segments_std[key] = first_image_cluster_segments_std[key] / first_image_cluster_ocurrence[key]
    
    second_image_cluster_segments_std = {}
    second_image_cluster_ocurrence = {}
    
    for index_1 in range(len(second_std_matrix)):
        for index_2 in range(len(second_std_matrix[index_1])):
            deviation_value = second_std_matrix[index_1][index_2]

            cluster_tuple = transitional_matrix[index_1][index_2]

            if cluster_tuple in second_image_cluster_segments_std:
                second_image_cluster_segments_std[cluster_tuple] = second_image_cluster_segments_std[cluster_tuple] + deviation_value
                second_image_cluster_ocurrence[cluster_tuple] = second_image_cluster_ocurrence[cluster_tuple] + 1
            else:
                second_image_cluster_segments_std[cluster_tuple] = deviation_value
                second_image_cluster_ocurrence[cluster_tuple] = 1

    for key in second_image_cluster_segments_std:
        second_image_cluster_segments_std[key] = second_image_cluster_segments_std[key] / second_image_cluster_ocurrence[key]
    
    return [first_image_cluster_segments_std, second_image_cluster_segments_std]


def merge_matrixes(first_label_matrix, second_label_matrix):
    final_matrix = np.empty((len(first_label_matrix), len(first_label_matrix[0])), dtype=object)
    
    for index_1 in range(len(first_label_matrix)):
        for index_2 in range(len(first_label_matrix[index_1])):
            final_matrix[index_1][index_2] = str(str(first_label_matrix[index_1][index_2]) + "-"+ str(second_label_matrix[index_1][index_2]))
            
    return final_matrix


def image_std(image):
    image_width, image_height = pil_image.size
    first_image = np.asarray(pil_image)
    image = pl.mean(first_image, 2)
    
    box_height = box_width = 3
    
    new_image = np.zeros(shape=(len(image),len(image[0])))
    
    for index_y in range(int(image_height/box_height)):
        for index_x in range(int(image_width/box_width)):
            
            points_array = [image[index_y * box_height, index_x * box_width], image[index_y * box_height,index_x * box_width + 1] , image[index_y * box_height + 1, index_x * box_width] , image[index_y * box_height + 1, index_x * box_width + 1], image[index_y * box_height + 2, index_x * box_width + 1], image[index_y * box_height + 1, index_x * box_width + 2], image[index_y * box_height, index_x * box_width + 2], image[index_y * box_height + 2, index_x * box_width], image[index_y * box_height + 2, index_x * box_width + 2]]
            points_std = np.std(points_array)

            new_image[index_y * box_height, index_x * box_width] = points_std
            new_image[index_y * box_height, index_x * box_width + 1] = points_std
            new_image[index_y * box_height + 1, index_x * box_width] = points_std
            new_image[index_y * box_height + 1, index_x * box_width + 1] = points_std
            new_image[index_y * box_height + 1, index_x * box_width + 2] = points_std
            new_image[index_y * box_height + 2, index_x * box_width + 1] = points_std
            new_image[index_y * box_height + 2, index_x * box_width + 2] = points_std
            new_image[index_y * box_height + 2, index_x * box_width + 0] = points_std
            new_image[index_y * box_height + 0, index_x * box_width + 2] = points_std
            # points_array1 = []
            # for box_height_index in range(box_height):
            #    for box_width_index in range(box_width):
            #        points_array1.append(image[index_y * box_height + box_height_index, index_x * box_width_index])
            # 
            # points_std = np.std(points_array1)
            # 
            # for box_height_index in range(box_height):
            #    for box_width_index in range(box_width):
            #        new_image1[index_y * box_height + box_height_index, index_x * box_width + box_width_index] = points_std
            
            
    fig = pl.figure(figsize=(5, 5))
    pl.imshow(new_image)
    pl.xticks(())
    pl.yticks(())
    fig.show()
    
    sp.misc.imsave('std_test.png', new_image)
    return new_image


###############################################################################
## BEGINNING
###############################################################################
st = time.time()

# Quickshift clustering alg. parameters
kernel_size = 12
max_dist = 200
ratio = 0.05

#first_image_name = 'boat_1.jpeg'
first_image_name = str(sys.argv[1])

#second_image_name = 'boat_2.jpeg'
second_image_name = str(sys.argv[2])

#scale = 0.25 # from 0 to 1
scale = float(sys.argv[3])

###############################################################################
## FIRST IMAGE
###############################################################################
pil_image = Image.open(first_image_name).convert('RGB')
width, height = pil_image.size

if scale != 1:
    pil_image = pil_image.resize((int(scale*width), int(scale*height)), Image.ANTIALIAS)

first_std_image = image_std(pil_image)

first_image = np.asarray(pil_image)
first_image_final = np.copy(first_image)

# Compute clustering
print("Compute structured hierarchical clustering...")

#first_label = felzenszwalb(pil_image, scale=100, sigma=0.5, min_size=50)
#first_label = slic(pil_image, ratio=10, n_segments=250, sigma=1)
#first_image_edges = filter.canny(first_image, sigma=5)
first_label = quickshift(pil_image, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)

fig, ax = plt.subplots(1, 3)
fig.set_size_inches(8, 3, forward=True)
plt.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05, 0.05)
ax[0].imshow(mark_boundaries(first_image_final, first_label))
ax[0].set_title("1st Image")

first_image = pl.mean(first_image,2)

first_label = np.reshape(first_label, first_image.shape)

###############################################################################
#SECOND IMAGE
###############################################################################
pil_image = Image.open(second_image_name)

if scale != 1:
    pil_image = pil_image.resize((int(scale*width), int(scale*height)), Image.ANTIALIAS).convert('RGB')

second_image = np.asarray(pil_image)

second_std_image = image_std(pil_image)

second_image_final = np.copy(second_image)

print("Compute structured hierarchical clustering...")

#second_label = felzenszwalb(pil_image, scale=100, sigma=0.5, min_size=50)
#second_label = slic(pil_image, ratio=10, n_segments=250, sigma=1)
#second_image_edges = filter.canny(second_image, sigma=4)
second_label = quickshift(pil_image, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)

ax[1].imshow(mark_boundaries(second_image_final, second_label))
ax[1].set_title("2nd Image")

second_image = pl.mean(second_image,2)

second_label = np.reshape(second_label, second_image.shape)

###############################################################################
# POST PROCESSING
###############################################################################
transitional_matrix = merge_matrixes(first_label, second_label)

segmented_stds = create_segmented_cluster_std_estimate(first_std_image, second_std_image, transitional_matrix)
first_image_cluster_segments_std = segmented_stds[0]
second_image_cluster_segments_std = segmented_stds[1]

print "LIST OF STD() PER SEGMENT:"
print segmented_stds


## CREATE 2D REPRESENTATION OF THE STD() ON BOTH IMAGE'S PIXELS DIVIDED BY THE INTERSECTION OF THE CLUSTERS MATRIXES
first_image_cluster_segments_std_array = np.zeros(shape=(len(transitional_matrix),len(transitional_matrix[0])))
second_image_cluster_segments_std_array = np.zeros(shape=(len(transitional_matrix),len(transitional_matrix[0])))
for index_1 in range(len(transitional_matrix)):
    for index_2 in range(len(transitional_matrix[index_1])):
        transitional_tuple = transitional_matrix[index_1][index_2]
        first_image_cluster_segments_std_array[index_1][index_2] = first_image_cluster_segments_std[transitional_tuple]
        second_image_cluster_segments_std_array[index_1][index_2] = second_image_cluster_segments_std[transitional_tuple]


## PRINT STD() GRAPHS
fig = pl.figure(figsize=(5, 5))
pl.imshow(first_image_cluster_segments_std_array, cmap = cm.Greys_r)
pl.xticks(())
pl.yticks(())
fig.show()

fig = pl.figure(figsize=(5, 5))
pl.imshow(second_image_cluster_segments_std_array, cmap = cm.Greys_r)
pl.xticks(())
pl.yticks(())
fig.show()


#print label
print("Elapsed time: ", time.time() - st)
print("Number of pixels: ", second_label.size)
print("Number of clusters: " +str(np.unique(second_label).size) + "  " +  str(np.unique(first_label).size))

###############################################################################
# FINAL IMAGE
###############################################################################
final_image = np.copy(first_image_final)
test_image = np.copy(first_image)

for index_1 in range(len(final_image)):
    for index_2 in range(len(final_image[index_1])):
        
        first_image_delta = first_image_cluster_segments_std_array[index_1][index_2]
        second_image_delta = second_image_cluster_segments_std_array[index_1][index_2]
        
        #Compare modules
        if first_image_delta > second_image_delta:
            final_image[index_1][index_2] = first_image_final[index_1][index_2]
        else:
            final_image[index_1][index_2] = second_image_final[index_1][index_2]
            
            
# Adds 3rd image and print big graph:
ax[2].imshow(final_image)
ax[2].set_title("Result Image")
for a in ax:
    a.set_xticks(())
    a.set_yticks(())
plt.show()

sp.misc.imsave("output/" + str(width) + "_" + str(height) + "_" + str(kernel_size) + "_" + str(max_dist) + "_" + str(ratio) + '_final.png', final_image)

pl.show()