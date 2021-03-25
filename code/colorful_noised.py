import numpy as np
import matplotlib as mlt
from matplotlib import pyplot as plt
from PIL import Image
import skimage.transform
import random
import math
import time
import os

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

def read_data():
    add_1 = os.path.join(os.getcwd(), "data", "sample6.jpg")
    image1 = Image.open(add_3)
    return image1

def colorful_noised():
    rand=0.07#percent of noise pixel variables in each image layer.
    image2 = read_data()
    image2=np.asarray(image2)
    plt.imshow(image2)
    image_new2=np.copy(image2)

    original_image2=image_new2

    noise_image2=image_new2


    [l2,w2,h2]=image_new2.shape

    plt.imshow(original_image2)
    plt.title('Original image')
    plt.grid(None) 
    plt.axis('off')
    plt.show()



    noise2=np.random.randint(0, 101, (l2, w2, h2))

    noise_image2=np.copy(original_image2)

    for m in range(0,l2):
        for n in range(0,w2):
            for h in range(0,h2):
                if noise2[m,n,h]<101*rand:
                    random_number=round(random.uniform(0,255))
                    noise_image2[m,n]=random_number
                else:
                    noise_image2[m,n]=noise_image2[m,n]
    plt.imshow(noise_image2)
    plt.title('Noise image')
    plt.grid(None) 
    plt.axis('off')
    plt.show()
    return original_image2, noise_image2