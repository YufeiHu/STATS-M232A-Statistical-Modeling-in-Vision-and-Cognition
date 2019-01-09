# -*- coding: utf-8 -*-
import matplotlib as mplt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


font = {'weight':'bold', 'size':22}
mplt.rc('font', **font)


def polarize(A):
    width = A.shape[0]
    height = A.shape[1]
    A_tmp = np.zeros((height * width))
    f_tmp = np.zeros((height * width))
    
    k = 0
    for x in range(width):
        for y in range(height):
            A_tmp[k] = A[x, y]
            f_tmp[k] = (float(x) ** 2 + float(y) ** 2) ** 0.5
            k = k + 1    
    
    f_polar = np.array([x for x, _ in sorted(zip(f_tmp, A_tmp))])
    A_polar = np.array([y for _, y in sorted(zip(f_tmp, A_tmp))])
    
    return A_polar, f_polar
    

def discretize(A, f, step):
    f_tmp = np.copy(f)
    A_tmp = np.copy(A)
    f_dis = np.arange(f.min(), f.max(), step)
    A_dis = np.zeros(f_dis.shape[0]-1)
    
    ptr = 0
    for i in range(f_dis.shape[0]-1):
        f_max = f_dis[i+1]
        A_sum = 0
        num = 0
        while f_tmp[ptr] < f_max and ptr < A.shape[0]:
            A_sum += A_tmp[ptr]
            num += 1
            ptr += 1
        if num != 0:
            A_dis[i] = A_sum / num
    
    f_dis = np.delete(f_dis, -1)
    return A_dis, f_dis
        

def powerIntegration(A, f):
    A_square = np.square(A)
    i_max = int(f.max() / 2)
    S = np.zeros(i_max)
    for i in range(i_max):
        index = np.where((f >= i) & (f <= 2*i))
        S[i] = np.sum(A_square[index])
    return S
    
    
def rgb2gray(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])


img1 = mpimg.imread('../data/natural_scene_1.jpg')
img2 = mpimg.imread('../data/natural_scene_2.jpg')
img3 = mpimg.imread('../data/natural_scene_3.jpg')
img4 = mpimg.imread('../data/natural_scene_4.jpg')


img1 = rgb2gray(img1)
img2 = rgb2gray(img2)
img3 = rgb2gray(img3)
img4 = rgb2gray(img4)


img1_fft = np.fft.fft2(img1)
img2_fft = np.fft.fft2(img2)
img3_fft = np.fft.fft2(img3)
img4_fft = np.fft.fft2(img4)


A1 = np.absolute(img1_fft)
A2 = np.absolute(img2_fft)
A3 = np.absolute(img3_fft)
A4 = np.absolute(img4_fft)


A1_polar, f1_polar = polarize(A1)
A2_polar, f2_polar = polarize(A2)
A3_polar, f3_polar = polarize(A3)
A4_polar, f4_polar = polarize(A4)


A1_dis, f1_dis = discretize(A1_polar, f1_polar, 1.0)
A2_dis, f2_dis = discretize(A2_polar, f2_polar, 1.0)
A3_dis, f3_dis = discretize(A3_polar, f3_polar, 1.0)
A4_dis, f4_dis = discretize(A4_polar, f4_polar, 1.0)


# Question (1)
plt.figure()
plt.plot(np.log(f1_dis), np.log(A1_dis), lw=5, label='image 1')
plt.plot(np.log(f2_dis), np.log(A2_dis), lw=5, label='image 2')
plt.plot(np.log(f3_dis), np.log(A3_dis), lw=5, label='image 3')
plt.plot(np.log(f4_dis), np.log(A4_dis), lw=5, label='image 4')
plt.xlabel('Log of frequency')
plt.ylabel('Log of amplitude')
plt.xlim([0, 6])
plt.legend()
plt.grid(True)


# Question (2)
S1 = powerIntegration(A1_dis, f1_dis)
S2 = powerIntegration(A2_dis, f2_dis)
S3 = powerIntegration(A3_dis, f3_dis)
S4 = powerIntegration(A4_dis, f4_dis)


plt.figure()
plt.plot(S1, lw=5, alpha=0.8, label='image 1')
plt.plot(S2, lw=5, alpha=0.8, label='image 2')
plt.plot(S3, lw=5, alpha=0.8, label='image 3')
plt.plot(S4, lw=5, alpha=0.8, label='image 4')
plt.xlabel('f0')
plt.ylabel('Integration of power')
plt.xlim([100, 600])
plt.ylim([-1e13, 5e13])
plt.legend()
plt.grid(True)