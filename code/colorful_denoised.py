import numpy as np
import matplotlib as mlt
from matplotlib import pyplot as plt
from PIL import Image
import skimage.transform
import random
import math
import time

from code.psnr import psnr255
from code.colorful_noised import pad_with, colorful_noised

def colorful_denoised(lamda_g=0.5, lamda_f=0.5, n1=1, n2=1, n3=1, w1=1, w2=0, number_iterations=1):
    original_image2, noise_image2 = colorful_noised()
    plt.figure(figsize=[10,10])

    plt.subplot(2,2,1)
    plt.imshow(original_image2)
    plt.title('Original image')
    plt.grid(None) 
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.imshow(noise_image2)
    plt.title('Noise image')
    plt.grid(None) 
    plt.axis('off')


    #Gibbs Sampling:


    #lamda_g: Weight parameter for G
    #lamda_f: Weight parameter for F
    #n1=1: t_{D}
    #n2=1: t_{G}
    #n3=1: t_{F}

    #w1=1: Weight parameter for difference between Y[i,j] and its neighbors in same hidden layer.
    #w2=0: Weight parameter for difference between Y[i,j] and the corresponded x[i,j].



    imag=noise_image2
    original_imag=original_image2
    [l,w,h]=imag.shape
    imag_expand=np.zeros([l+2,w+2,h],dtype=np.uint8)
    for i in range(0,h):
        imag_expand[:,:,i]=np.pad(imag[:,:,i],1, pad_with, padder=255)

    imag_expand_non=np.copy(imag_expand)

    #number_iterations: number of circles for de-noising image.

    p=np.arange(256)
    E=np.copy(p)
    for i in range(0,number_iterations):
        for s in range(0,h):

            for m in range(1,l+1):
              #print(m)
                for n in range(1,w+1):
                    a=imag_expand[m,n,s]
                    a_neigh=np.array([imag_expand[m,n-1,s],imag_expand[m,n+1,s],imag_expand[m-1,n,s],imag_expand[m+1,n,s],imag_expand[m+1,n-1,s],imag_expand[m+1,n+1,s],imag_expand[m-1,n+1,s],imag_expand[m-1,n+1,s]])
                    a_neigh_list=abs(np.array([imag_expand[m,n-1,s]-imag_expand[m,n+1,s],imag_expand[m-1,n,s]-imag_expand[m+1,n,s],imag_expand[m-1,n-1,s]-imag_expand[m+1,n+1,s],imag_expand[m-1,n+1,s]-imag_expand[m+1,n-1,s]]))
                    a_neigh_list_ave=np.array([imag_expand[m,n-1,s]+imag_expand[m,n+1,s],imag_expand[m-1,n,s]+imag_expand[m+1,n,s],imag_expand[m-1,n-1,s]+imag_expand[m+1,n+1,s],imag_expand[m-1,n+1,s]+imag_expand[m+1,n-1,s]])/2
                    for i1 in range(0,256):
                        E[i1]=w2*abs(a-p[i1])**n1+w1*abs(np.sum((a_neigh-p[i1])**n1))+lamda_f*abs(a_neigh_list_ave[np.where(a_neigh_list==np.min(a_neigh_list))[0][0]]-p[i1])**n3+lamda_g*abs(np.mean(a_neigh)-p[i1])**n2
                    location=round(np.where(E==np.min(E))[0][0])
                    imag_expand[m,n,s]=location
                    
        imag_expand_non = np.copy(imag_expand)

    imag_expand_m=np.delete(imag_expand, [0,l+1], axis=0)
    imag_expand_m=np.delete(imag_expand_m, [0,w+1], axis=1)
    d=psnr255(imag_expand_m, original_image2)
    plt.subplot(2,2,3)
    plt.title('Gibbs Sampling: PSNR = %.4f dB'%d)
    plt.imshow(imag_expand_m)
    plt.grid(None) 
    plt.axis('off')



    #Midified ICM:


    #lamda_g: Weight parameter for G
    #lamda_f: Weight parameter for F
    #n1=1: t_{D}
    #n2=1: t_{G}
    #n3=1: t_{F}

    #w1=1: Weight parameter for difference between Y[i,j] and its neighbors in same hidden layer.
    #w2=0: Weight parameter for difference between Y[i,j] and the corresponded x[i,j].


    imag=noise_image2
    #print(imag[1,:])
    original_imag=original_image2
    [l,w,h]=imag.shape
    imag_expand=np.zeros([l+2,w+2,h],dtype=np.uint8)
    for i in range(0,h):
        imag_expand[:,:,i]=np.pad(imag[:,:,i],1, pad_with, padder=255)

    imag_expand_non=np.copy(imag_expand)
    #print(imag_expand)

    #number_iterations: number of circles for de-noising image.

    p=np.arange(256)
    E=np.copy(p)
    for i in range(0,number_iterations):
        for s in range(0,h):

            for m in range(1,l+1):
              #print(m)
                for n in range(1,w+1):
                    a=imag_expand_non[m,n,s]
                    a_neigh=np.array([imag_expand_non[m,n-1,s],imag_expand_non[m,n+1,s],imag_expand_non[m-1,n,s],imag_expand_non[m+1,n,s],imag_expand_non[m+1,n-1,s],imag_expand_non[m+1,n+1,s],imag_expand_non[m-1,n+1,s],imag_expand_non[m-1,n+1,s]])
                    a_neigh_list=abs(np.array([imag_expand_non[m,n-1,s]-imag_expand_non[m,n+1,s],imag_expand_non[m-1,n,s]-imag_expand_non[m+1,n,s],imag_expand_non[m-1,n-1,s]-imag_expand_non[m+1,n+1,s],imag_expand_non[m-1,n+1,s]-imag_expand_non[m+1,n-1,s]]))
                    a_neigh_list_ave=np.array([imag_expand_non[m,n-1,s]+imag_expand_non[m,n+1,s],imag_expand_non[m-1,n,s]+imag_expand_non[m+1,n,s],imag_expand_non[m-1,n-1,s]+imag_expand_non[m+1,n+1,s],imag_expand_non[m-1,n+1,s]+imag_expand_non[m+1,n-1,s]])/2
                    for i1 in range(0,256):
                        E[i1]=w2*abs(a-p[i1])**n1+w1*abs(np.sum((a_neigh-p[i1])**n1))+lamda_f*abs(a_neigh_list_ave[np.where(a_neigh_list==np.min(a_neigh_list))[0][0]]-p[i1])**n3+lamda_g*abs(np.mean(a_neigh)-p[i1])**n2
                    location=round(np.where(E==np.min(E))[0][0])
                    imag_expand[m,n,s]=location

    imag_expand_m=np.delete(imag_expand, [0,l+1], axis=0)
    imag_expand_m=np.delete(imag_expand_m, [0,w+1], axis=1)
    noise_image111=np.copy(imag_expand_m)
    d=psnr255(imag_expand_m, original_image2)
    plt.subplot(2,2,4)
    plt.imshow(imag_expand_m)
    plt.title('Modified ICM: PSNR = %.4f dB'%d)
    plt.grid(None) 
    plt.axis('off')
    plt.show()