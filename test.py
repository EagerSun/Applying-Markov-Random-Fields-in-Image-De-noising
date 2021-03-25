from code.binary_denoised import binary_denoised
from code.colorful_denoised import colorful_denoised

#Denoised the image: sample1.jpg
binary_denoised(sample = "sample1", lamda_g=1, lamda_f=1, n1=2, n2=1, n3=1, w1=1, w2=1, number_iterations=1)

#Denoised the image: sample3.jpg
binary_denoised(sample = "sample3", lamda_g=1, lamda_f=1, n1=2, n2=1, n3=1, w1=1, w2=1, number_iterations=1)

#Denoised the image: sample6.jpg
colorful_denoised(lamda_g=0.5, lamda_f=0.5, n1=1, n2=1, n3=1, w1=1, w2=0, number_iterations=1)