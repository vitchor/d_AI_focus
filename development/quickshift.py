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
#from skimage import filter

def image_std(image):
    image_width, image_height = pil_image.size
    
    first_image = np.asarray(pil_image)
    image = pl.mean(first_image, 2)
    
    box_height = box_width = 3
    
    new_image = np.copy(image)
    
    for index_y in range(int(image_height/box_height)):
        for index_x in range(int(image_width/box_width)):
            print 'idexes'
            print index_x
            print index_y
            
            
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
            #        
            # 
            # ###
            
     
            
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
      
            #print points_array1
            print points_array
            # print new_image2 == new_image1
    fig = pl.figure(figsize=(5, 5))
    pl.imshow(new_image)
    pl.xticks(())
    pl.yticks(())
    fig.show()
    
    sp.misc.imsave('std_test.png', new_image)
    return new_image


#first_image_name = 'boat_1.jpeg'
first_image_name = str(sys.argv[1])

#second_image_name = 'boat_2.jpeg'
second_image_name = str(sys.argv[2])

#scale = 0.25 # from 0 to 1
scale = float(sys.argv[3])

#n_clusters = 8  # number of regions
n_clusters = int(sys.argv[4])


pil_image = Image.open(first_image_name).convert('RGB')
width, height = pil_image.size

if scale != 1:
    pil_image = pil_image.resize((int(scale*width), int(scale*height)), Image.ANTIALIAS)


first_std_image = image_std(pil_image)

first_image = np.asarray(pil_image)
first_image_final = np.copy(first_image)


#first_image_edges = filter.canny(first_image, sigma=5)

#X = np.reshape(first_image, (-1, 1))
#connectivity = grid_to_graph(*first_image.shape)

###############################################################################
# Compute clustering
print("Compute structured hierarchical clustering...")
st = time.time()

#first_label = felzenszwalb(pil_image, scale=100, sigma=0.5, min_size=50)
#first_label = slic(pil_image, ratio=10, n_segments=250, sigma=1)
first_label = quickshift(pil_image, kernel_size=10, max_dist=80, ratio=0.1)
#ward = Ward(n_clusters=n_clusters, connectivity=connectivity, compute_full_tree=False).fit(X)

fig, ax = plt.subplots(1, 3)
fig.set_size_inches(8, 3, forward=True)
plt.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05, 0.05)

ax[0].imshow(mark_boundaries(first_image_final, first_label))
ax[0].set_title("1st Image")


first_image_cluster_number = len(np.unique(first_label))

first_image = pl.mean(first_image,2)

first_label = np.reshape(first_label, first_image.shape)

first_image_cluster_deltas = []

for cluster_index in range(first_image_cluster_number):
    
    #new_image = np.copy(first_image)
    cluster_points = []
    
    for index_1 in range(len(first_label)):
        for index_2 in range(len(first_label[index_1])):
            
            if first_label[index_1][index_2] == cluster_index:
                cluster_points.append(first_std_image[index_1][index_2])
                
            
    
    #histogram = np.histogram(cluster_points)
    #splitted_arrays = np.array_split(histogram[0], 4)
    
    #maxValue = max([max(splitted_arrays[0]), max(splitted_arrays[3])])
    #minValue = min([min(splitted_arrays[1]), min(splitted_arrays[2])])
    
    if len(cluster_points) > 0:
        average = sum(cluster_points)/len(cluster_points)
        #delta = np.std(cluster_points) #/len(cluster_points)# / average
        delta = average
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

pil_image = pil_image.resize((int(scale*width), int(scale*height)), Image.ANTIALIAS).convert('RGB')

second_image = np.asarray(pil_image)

second_std_image = image_std(pil_image)

second_image_final = np.copy(second_image)
#second_image = pl.mean(second_image,2)

#second_image_edges = filter.canny(second_image, sigma=4)

#X = np.reshape(second_image, (-1, 1))


#X = np.reshape(second_image, (-1, 1))


# Define the structure A of the data. Pixels connected to their neighbors.
#connectivity = grid_to_graph(*second_image.shape)

# Compute clustering
print("Compute structured hierarchical clustering...")
st = time.time()



#second_label = np.copy(second_image)

#second_label_3d = ICM(second_image_final, 2, 8)

# for label_index_1 in range(len(second_label_3d)):
#     for label_index_2 in range(len(second_label_3d[label_index_1])):
#         
#         row = second_label_3d[label_index_1][label_index_2]
#         second_label[label_index_1][label_index_2] = int(row[0] * 1 + row[1] * 2 + row[2] * 4 + row[3] * 8)

#ward = Ward(n_clusters=n_clusters, connectivity=connectivity, compute_full_tree=False).fit(X)
#second_label = np.reshape(ward.labels_, second_image.shape)

#second_label = felzenszwalb(pil_image, scale=100, sigma=0.5, min_size=50)
#second_label = slic(pil_image, ratio=10, n_segments=250, sigma=1)
second_label = quickshift(pil_image, kernel_size=10, max_dist=80, ratio=0.1)
#ward = Ward(n_clusters=n_clusters, connectivity=connectivity, compute_full_tree=False).fit(X)

ax[1].imshow(mark_boundaries(second_image_final, second_label))
ax[1].set_title("2nd Image")

second_image_cluster_number = len(np.unique(second_label))

second_image = pl.mean(second_image,2)

second_label = np.reshape(second_label, second_image.shape)

second_image_cluster_deltas = []

for cluster_index in range(second_image_cluster_number):

    #new_image = np.copy(second_image)
    cluster_points = []

    for index_1 in range(len(second_label)):
        for index_2 in range(len(second_label[index_1])):
            if second_label[index_1][index_2] == cluster_index:
                cluster_points.append(second_std_image[index_1][index_2])
                


    #histogram = np.histogram(cluster_points)
    #splitted_arrays = np.array_split(histogram[0], 4)

    #maxValue = max([max(splitted_arrays[0]), max(splitted_arrays[3])])
    #minValue = min([min(splitted_arrays[1]), min(splitted_arrays[2])])
    
    print "aqui" 
    print len(cluster_points)
    #delta = (maxValue - minValue) * (max(histogram[1]) - min(histogram[1]))
    
    if len(cluster_points) > 0:
        average = sum(cluster_points)/len(cluster_points)
        #delta = np.std(cluster_points)#/len(cluster_points) # / average
        delta = average
    else:
        delta = 0
        
    print delta
    
    second_image_cluster_deltas.append(delta)
    
    #sp.misc.imsave(second_image_name +"_" + str(len(cluster_points)) + "_" + str(cluster_index) + '_autosave.png', new_image)

# pl.figure(figsize=(5, 5))
# pl.imshow(first_image_final)
# for l in range(n_clusters):
#    pl.contour(first_label == l, contours=1,
#              colors=[pl.cm.spectral(l / float(n_clusters)), ])
# pl.xticks(())
# pl.yticks(())

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
            
        
        
# pl.figure(figsize=(5, 5))
# pl.imshow(second_image_final, cmap=pl.cm.gray)
# for l in range(n_clusters):
#    pl.contour(second_label == l, contours=1,
#               colors=[pl.cm.spectral(l / float(n_clusters)), ])
# pl.xticks(())
# pl.yticks(())

#final_image = ndI.median_filter(final_image, 2)
#from scikits.image.filter import tv_denoise
#from tv_denoise import tv_denoise
#final_image = denoise_tv_chambolle(final_image, weight=0.1, multichannel=True)

ax[2].imshow(final_image)
ax[2].set_title("Result Image")
for a in ax:
    a.set_xticks(())
    a.set_yticks(())
plt.show()

sp.misc.imsave("output/" + second_image_name + "_" + str(cluster_index + 1) + '_final.png', final_image)

pl.show()

