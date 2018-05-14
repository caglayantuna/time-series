import siamxt
from project_functions import *
from scipy.misc import imsave

Image = geoimread('samples/kalideosdata/S2dataset2017.tif')

imarray=geoImToArray(Image)
imarray= imarray.clip(min=0)
imarray=imarray.astype(np.uint8)

distance_img=distanceSITS(imarray)


#Tree construction

w,h=distance_img.shape
result=np.zeros([w,h],dtype=float)
for x in range(w):
    for y in range(h):
        if distance_img[x,y]>=(distance_img.mean()+40):
            result[x, y]=255
        else:
            result[x, y]=0


Bc = np.ones((3,3),dtype = bool)
input_img=np.array(result, dtype=np.uint8)
mxt = siamxt.MaxTreeAlpha(input_img,Bc)

result2=attribute_area_filter(mxt,(140))
imsave('samples/kalideosdata/kalideoschangedistance.png', result2)
