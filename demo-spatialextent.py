import siamxt
from project_functions import *
from sklearn.cluster import KMeans
from scipy.misc import imsave


def tree_one(imarray,r,c):

    Bc = np.ones((3, 3), dtype=bool)

    mxt = siamxt.MaxTreeAlpha(imarray, Bc)
    attributes1= attribute_area_filter(mxt, (500))
    attributes2= attribute_area_filter(mxt, (5000))
    attributes3= attribute_area_filter(mxt, (50000))
    attributes=np.stack((imarray,attributes1,attributes2,attributes3),axis=-1)
    image_array = np.reshape(attributes, (r * c, 4))
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
     mean_img = meanSITS(imarray)

     r, c= tuple(mean_img.shape)
     image_array=tree_one(mean_img,r,c)
     result=kmeans_cluster(image_array,6,r,c)
     imsave('samples/jordandataset/SPextenttreemean_6class_jordan.png', result)


