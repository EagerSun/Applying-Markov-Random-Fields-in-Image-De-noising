import numpy 
import math
import cv2
def psnr1(img1, img2):# for binary images cases
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    else:
      
        PIXEL_MAX = 1.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def psnr255(img1, img2):# for colored image case
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    else:
      
        PIXEL_MAX = 255
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))