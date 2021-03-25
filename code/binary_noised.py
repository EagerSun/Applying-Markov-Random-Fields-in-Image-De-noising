import os
import numpy as np
import matplotlib as mlt
from matplotlib import pyplot as plt
from PIL import Image
import skimage.transform
import random
import math
import time

def read_images():
    add_1 = os.path.join(os.getcwd(), "data", "sample1.jpg")
    add_3 = os.path.join(os.getcwd(), "data", "sample3.jpg")
    image1, image3 = Image.open(add_1), Image.open(add_3)
    return image1, image3

def binary_noised():
    rand=0.2#percent of noise pixels in all pixels.
    
    image1, image3 = read_images()
    image1=np.asarray(image1)
    image3=np.asarray(image3)

    image_new1=skimage.transform.resize(image1[:,:,0], (500,866))
    image_new3=skimage.transform.resize(image3[:,:,0], (300,300))

    original_image1=image_new1
    original_image3=image_new3

    noise_image1=image_new1
    noise_image3=image_new3


    [l1,w1]=image_new1.shape
    [l3,w3]=image_new3.shape

    for m in range(0,l1):
        for n in range(0,w1):
            if image_new1[m,n]==0:
                original_image1[m,n]=0
            else:
                original_image1[m,n]=1
    plt.imshow(original_image1)
    plt.title('Original image')
    plt.grid(None) 
    plt.axis('off')
    plt.show()

    for m in range(0,l3):
        for n in range(0,w3):
            if image_new3[m,n]==0:
                original_image3[m,n]=0
            else:
                original_image3[m,n]=1
    plt.imshow(original_image3)
    plt.title('Original image')
    plt.grid(None) 
    plt.axis('off')
    plt.show()


    noise1=np.random.randint(0, 101, (l1, w1))
    noise3=np.random.randint(0, 101, (l3, w3))

    noise_image1=np.copy(original_image1)
    noise_image3=np.copy(original_image3)

    for m in range(0,l1):
        for n in range(0,w1):
            if noise1[m,n]<101*rand:
                random_number=random.uniform(0,1)
                noise_image1[m,n]=random_number
            else:
                noise_image1[m,n]=noise_image1[m,n]
    plt.imshow(noise_image1)
    plt.title('Noise image')
    plt.grid(None) 
    plt.axis('off')
    plt.show()

    for m in range(0,l3):
        for n in range(0,w3):
            if noise3[m,n]<101*rand:
                random_number=random.uniform(0,1)
                noise_image3[m,n]=random_number
            else:
                noise_image3[m,n]=noise_image3[m,n]
    plt.imshow(noise_image3)
    plt.title('Noise image')
    plt.grid(None) 
    plt.axis('off')
    plt.show()
    
    return (original_image1, noise_image1), (original_image3, noise_image3)