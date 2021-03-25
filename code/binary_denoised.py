import numpy as np
import matplotlib as mlt
from matplotlib import pyplot as plt
from PIL import Image
import skimage.transform
import random
import math
import time

from code.binary_noised import binary_noised
from code.psnr import psnr1

def binary_denoised(sample = "sample3", lamda_g=1, lamda_f=1, n1=2, n2=1, n3=1, w1=1, w2=1, number_iterations=1):
    
    if sample == "sample3":
        (_, _), (original_image1, noise_image1) = binary_noised()
        
    elif sample == "sample1":
        (original_image1, noise_image1), (_, _) = binary_noised()
    #lamda_g: Weight parameter for G
    #lamda_f=1: Weight parameter for F
    #n1=2: t_{D}
    #n2=1: t_{G}
    #n3=1: t_{F}

    #w1=1: Weight parameter for difference between Y[i,j] and its neighbors in same hidden layer.
    #w2=1: Weight parameter for difference between Y[i,j] and the  corresponded x[i,j].

    plt.figure(figsize=[20,10])

    plt.subplot(2,2,1)
    plt.imshow(original_image1)
    plt.title('Original image')
    plt.grid(None) 
    plt.axis('off')

    plt.subplot(2,2,2)
    plt.imshow(noise_image1)
    plt.title('Noise image')
    plt.grid(None) 
    plt.axis('off')


    #Modified ICM:

    imag=noise_image1
    #print(imag[1,:])
    original_imag=original_image1
    [l,w]=imag.shape
    imag_expand=np.lib.pad(imag,((1,1),(1,1)),'constant',constant_values=(0))
    original_imag=np.lib.pad(original_imag,((1,1),(1,1)),'constant',constant_values=(0))
    #print(imag_expand)
    imag_expand_unchange=imag_expand
    imag_expand_non=np.copy(imag_expand)

    # number_iterations: number of circles for de-noising image.


    for i in range(0,number_iterations):
        for m in range(1,l+1):
            for n in range(1,w+1):
                a=imag_expand_non[m,n]
                a_neigh=np.array([imag_expand_non[m,n-1],imag_expand_non[m,n+1],imag_expand_non[m-1,n],imag_expand_non[m+1,n],imag_expand_non[m+1,n-1],imag_expand_non[m+1,n+1],imag_expand_non[m-1,n+1],imag_expand_non[m-1,n+1]])
                a_neigh_list=abs(np.array([imag_expand_non[m,n-1]-imag_expand_non[m,n+1],imag_expand_non[m-1,n]-imag_expand_non[m+1,n],imag_expand_non[m-1,n-1]-imag_expand_non[m+1,n+1],imag_expand_non[m-1,n+1]-imag_expand_non[m+1,n-1]]))
                a_neigh_list_ave=np.array([imag_expand_non[m,n-1]+imag_expand_non[m,n+1],imag_expand_non[m-1,n]+imag_expand_non[m+1,n],imag_expand_non[m-1,n-1]+imag_expand_non[m+1,n+1],imag_expand_non[m-1,n+1]+imag_expand_non[m+1,n-1]])/2

                E1=w2*abs(a-1)**n1+w1*abs(np.sum((a_neigh-1)**n1))+lamda_g*abs(np.mean(a_neigh)-1)**n2+lamda_f*abs(a_neigh_list_ave[np.where(a_neigh_list==np.min(a_neigh_list))[0][0]]-1)**n3
                E0=w2*abs(a-0)**n1+w1*abs(np.sum((a_neigh-0)**n1))+lamda_g*abs(np.mean(a_neigh)-0)**n2+lamda_f*abs(a_neigh_list_ave[np.where(a_neigh_list==np.min(a_neigh_list))[0][0]]-0)**n3
                if E1>E0:
                    imag_expand[m,n]=0
                else:
                    imag_expand[m,n]=1
                    
        imag_expand_non = np.copy(imag_expand)


    imag_expand_m=np.delete(imag_expand, [0,l+1], axis=0)
    imag_expand_m=np.delete(imag_expand_m, [0,w+1], axis=1)
    d=psnr1(original_image1,imag_expand_m)
    plt.subplot(2,2,3)
    plt.imshow(imag_expand_m)
    plt.grid(None)
    plt.title('Modified ICM: PSNR = %.4f dB'%d)
    plt.axis('off')


    #Gibbs Sampling:


    imag=noise_image1
    #print(imag[1,:])
    original_imag=original_image1
    [l,w]=imag.shape
    imag_expand=np.lib.pad(imag,((1,1),(1,1)),'constant',constant_values=(0))
    original_imag=np.lib.pad(original_imag,((1,1),(1,1)),'constant',constant_values=(0))
    #print(imag_expand)
    imag_expand_unchange=imag_expand
    imag_expand_non=np.copy(imag_expand)

    # number_iterations: number of circles for de-noising image.


    for i in range(0,number_iterations):
        for m in range(1,l+1):
            for n in range(1,w+1):
                a=imag_expand[m,n]
                a_neigh=np.array([imag_expand[m,n-1],imag_expand[m,n+1],imag_expand[m-1,n],imag_expand[m+1,n],imag_expand[m+1,n-1],imag_expand[m+1,n+1],imag_expand[m-1,n+1],imag_expand[m-1,n+1]])
                a_neigh_list=abs(np.array([imag_expand[m,n-1]-imag_expand[m,n+1],imag_expand[m-1,n]-imag_expand[m+1,n],imag_expand[m-1,n-1]-imag_expand[m+1,n+1],imag_expand[m-1,n+1]-imag_expand[m+1,n-1]]))
                a_neigh_list_ave=np.array([imag_expand[m,n-1]+imag_expand[m,n+1],imag_expand[m-1,n]+imag_expand[m+1,n],imag_expand[m-1,n-1]+imag_expand[m+1,n+1],imag_expand[m-1,n+1]+imag_expand[m+1,n-1]])/2

                E1=w2*abs(a-1)**n1+w1*abs(np.sum((a_neigh-1)**n1))+lamda_g*abs(np.mean(a_neigh)-1)**n2+lamda_f*abs(a_neigh_list_ave[np.where(a_neigh_list==np.min(a_neigh_list))[0][0]]-1)**n3
                E0=w2*abs(a-0)**n1+w1*abs(np.sum((a_neigh-0)**n1))+lamda_g*abs(np.mean(a_neigh)-0)**n2+lamda_f*abs(a_neigh_list_ave[np.where(a_neigh_list==np.min(a_neigh_list))[0][0]]-0)**n3
                if E1>E0:
                    imag_expand[m,n]=0
                else:
                    imag_expand[m,n]=1


    imag_expand_m=np.delete(imag_expand, [0,l+1], axis=0)
    imag_expand_m=np.delete(imag_expand_m, [0,w+1], axis=1)
    d=psnr1(original_image1,imag_expand_m)
    plt.subplot(2,2,4)
    plt.imshow(imag_expand_m)
    plt.grid(None)
    plt.title('Gibbs Sampling: PSNR = %.4f dB'%d) 
    plt.axis('off')
    plt.show()