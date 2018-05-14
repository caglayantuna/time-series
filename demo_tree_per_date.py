import siamxt
from project_functions import *
import numpy as np
from scipy.misc import imsave
from sklearn.cluster import KMeans

def tree_per_date(imarray,r,c,b):

    Bc = np.ones((3, 3), dtype=bool)
    attributes = np.zeros([r, c, 3 * b], dtype=float)
    for i in range(b):
        mxt = siamxt.MaxTreeAlpha(imarray[:, :, i], Bc)
        attributes[:, :, i] = attribute_area_filter(mxt, (500))
        attributes[:, :, b + i] = attribute_area_filter(mxt, (5000))
        attributes[:, :, 2 * b + i] = attribute_area_filter(mxt, (50000))

    image_array = np.reshape(attributes, (r * c, 3 * b))
    return image_array

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
     image_array=tree_per_date(imarray,r,c,b)
     result=kmeans_cluster(image_array,5,r,c)
     imsave('samples/jordandataset/denemetpdmax_5class_jordan.png', result)

