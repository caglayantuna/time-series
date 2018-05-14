import siamxt
from project_functions import *
from scipy.misc import imsave

Image = geoimread('samples/spaindataset/spaingraymerge.tif')

imarray=geoImToArray(Image)
imarray= imarray.clip(min=0)
imarray=imarray.astype(np.uint8)

std_img=stdSITS(imarray)


#Tree construction
Bc = np.ones((3,3),dtype = bool)
input_img=np.array(std_img, dtype=np.uint8)
mxt = siamxt.MaxTreeAlpha(input_img,Bc)

result=attribute_area_filter(mxt,(1000))
w,h=result.shape
for x in range(w):
    for y in range(h):
        if result[x,y]>=result.mean():
            result[x, y]=255
        else:
            result[x, y]=0

#write geotiff
imsave('samples/jordandataset/jordan_stdfilter1000.png', result)

