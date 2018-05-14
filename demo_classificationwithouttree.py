from sklearn.cluster import KMeans
from project_functions import *
import numpy as np
from scipy.misc import imsave

def kmeans_cluster(image_array,n,r,c):

    y_pred = KMeans(n_clusters=n).fit_predict(image_array)
    colours = [[255, 0, 255], [0, 0, 0], [255, 0, 0], [0, 255, 0],
               [255, 255, 0], [0, 255, 255], [0, 0, 255]]

    y_pred_image = np.reshape(y_pred, (r, c))
    result = np.zeros([r, c, 3], dtype=float)
    for x in range(r):
        for y in range(c):
            col = y_pred_image[x, y]
            result[x, y] = colours[col]

    result = np.array(result, dtype=np.uint8)
    return result

if __name__ == "__main__":

     Image = geoimread('samples/jordandataset/jordangraymerge.tif')

     imarray=geoImToArray(Image)
     imarray=imarray.astype(np.uint16)
     r, c, b = tuple(imarray.shape)
     image_array = np.reshape(imarray, (r * c, b))

     result=kmeans_cluster(image_array,6,r,c)
     imsave('samples/jordandataset/Withouttree_6class_jordan.png', result)



