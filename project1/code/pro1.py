# -*- coding: utf-8 -*-
import matplotlib as mplt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
import scipy.misc
import scipy.stats
import numpy as np


font = {'weight':'bold', 'size':22}
mplt.rc('font', **font)


def gradientHorizontal(img):
    imgx = np.copy(img)
    imgx[:, :-1] = imgx[:, 1:] - imgx[:, :-1]
    imgx[:, -1] = 0
    return imgx


def gradientVertical(img):
    return np.rot90(gradientHorizontal(np.rot90(img)), 3)


def rgb2gray(img):
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])


def grayRescale(img):
    img_rescale = np.copy(img)
    img_rescale = img_rescale / 8.
    img_rescale = np.floor(img_rescale)
    return img_rescale
    

# ===================== Read and preprocess original image ====================
img = mpimg.imread('../data/natural_scene_1.jpg')
img = rgb2gray(img)
img = grayRescale(img)


# ===================== Compute the gradients of the image ====================
num_bin = 50
imgx = gradientHorizontal(img)
imgy = gradientVertical(img)


# Extra
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(img, cmap = plt.get_cmap('gray'))
plt.title('Original Image')
plt.xticks([])
plt.yticks([])
plt.subplot(1, 3, 2)
plt.imshow(imgx, cmap = plt.get_cmap('gray'))
plt.title('Horizontal Gradients')
plt.xticks([])
plt.yticks([])
plt.subplot(1, 3, 3)
plt.imshow(imgy, cmap = plt.get_cmap('gray'))
plt.title('Vertical Gradients')
plt.xticks([])
plt.yticks([])


# Question (1),(2)
plt.figure()
imgx_flat = imgx.flatten()
plt.hist(imgx_flat, num_bin, normed=True)
mean = np.sum(imgx_flat) / imgx_flat.shape[0]
var = np.var(imgx_flat)
std = np.std(imgx_flat)
kur = scipy.stats.kurtosis(imgx_flat)
plt.xlabel('Bins of gradient differences')
plt.ylabel('Frequency')
title_str = ('Histogram of horizontal gradients: mean={0:.4f}, variance={1:.4f}, kurtosis={2:.4f}'
             .format(mean, var, kur))
plt.title(title_str)
plt.grid(True)


plt.figure()
plt.hist(imgx_flat, num_bin, log=True, normed=True)
plt.xlabel('Bins of gradient differences')
plt.ylabel('Log of frequency')
plt.title('Log histogram of horizontal gradients')
plt.grid(True)


# Question (3)
img_GGD = mpimg.imread('../data/natural_scene_1.jpg')
img_GGD = rgb2gray(img_GGD)
imgx_GGD = gradientHorizontal(img_GGD)
imgx_flat_GGD = imgx_GGD.flatten()

GGD_param = scipy.stats.gennorm.fit(imgx_flat_GGD, floc=0)
GGD_x = np.linspace(imgx_flat_GGD.min(), imgx_flat_GGD.max(), imgx_flat_GGD.shape[0])
GGD_curve = scipy.stats.gennorm.pdf(GGD_x, *GGD_param)
plt.figure()
plt.hist(imgx_flat_GGD, 800, normed=True)
plt.plot(GGD_x, GGD_curve, 'r--', lw=5, alpha=0.6, label='gennorm pdf')
plt.xlabel('Bins of gradient differences')
plt.ylabel('Frequency')
plt.title('Histogram of horizontal gradients with fitted GGD curve')
plt.grid(True)


# Question (4)
NGD_param = scipy.stats.norm.fit(imgx_flat, floc=0)
NGD_x = np.linspace(imgx_flat.min(), imgx_flat.max(), imgx_flat.shape[0])
NGD_curve = scipy.stats.norm.pdf(NGD_x, *NGD_param)
plt.figure()
plt.hist(imgx_flat, num_bin, normed=True)
plt.plot(NGD_x, NGD_curve, 'r--', lw=5, alpha=0.6, label='norm pdf')
plt.xlabel('Bins of gradient differences')
plt.ylabel('Frequency')
plt.title('Histogram of horizontal gradients with fitted normal distribution curve')
plt.grid(True)


NGD_curve_log = np.log10(NGD_curve)
plt.figure()
plt.hist(imgx_flat, num_bin, log=True, normed=1)
plt.plot(NGD_x, NGD_curve, 'r--', lw=5, alpha=0.6, label='log norm pdf')
plt.xlabel('Bins of gradient differences')
plt.ylabel('Log of frequency')
plt.title('Log histogram of horizontal gradients with fitted normal distribution curve')
plt.grid(True)


# Question (5)
img_down1 = scipy.misc.imresize(img, (int(img.shape[0]/2), int(img.shape[1]/2)), 'bilinear')
img_down1 = grayRescale(img_down1)
imgx_down1 = gradientHorizontal(img_down1)

img_down2 = scipy.misc.imresize(img_down1, (int(img_down1.shape[0]/2), int(img_down1.shape[1]/2)), 'bilinear')
img_down2 = grayRescale(img_down2)
imgx_down2 = gradientHorizontal(img_down2)

img_down3 = scipy.misc.imresize(img_down2, (int(img_down2.shape[0]/2), int(img_down2.shape[1]/2)), 'bilinear')
img_down3 = grayRescale(img_down3)
imgx_down3 = gradientHorizontal(img_down3)


plt.figure()
plt.hist(imgx_flat, num_bin, normed=True, label='original', lw=3, fc=(0, 0, 1, 0.6))
plt.hist(imgx_down1.flatten(), 30, normed=True, label='down-sampled (1 time)', lw=3, fc=(1, 0, 0, 0.6))
plt.hist(imgx_down3.flatten(), 25, normed=True, label='down-sampled (3 time)', lw=3, fc=(0, 1, 1, 0.6))
plt.xlabel('Bins of gradient differences')
plt.ylabel('Frequency')
plt.title('Histogram of horizontal gradients (down-sampled 1 time)')
plt.legend()
plt.grid(True)


plt.figure()
plt.hist(imgx_flat, num_bin, normed=True, log=True, label='original', lw=3, fc=(0, 0, 1, 0.6))
plt.hist(imgx_down1.flatten(), 30, normed=True, log=True, label='down-sampled (1 time)', lw=3, fc=(1, 0, 0, 0.6))
plt.hist(imgx_down3.flatten(), 25, normed=True, log=True, label='down-sampled (3 time)', lw=3, fc=(0, 1, 1, 0.6))
plt.xlabel('Bins of gradient differences')
plt.ylabel('Log of Frequency')
plt.title('Log histogram of horizontal gradients (down-sampled 1 time)')
plt.legend()
plt.grid(True)


# Question (6)
img_rand = np.random.rand(img.shape[0], img.shape[1]) * 32
img_rand = np.floor(img_rand)
imgx_rand = gradientHorizontal(img_rand)


imgx_flat_rand = imgx_rand.flatten()
NGD_param_rand = scipy.stats.norm.fit(imgx_flat_rand, floc=0)
NGD_x_rand = np.linspace(imgx_flat_rand.min(), imgx_flat_rand.max(), imgx_flat_rand.shape[0])
NGD_curve_rand = scipy.stats.norm.pdf(NGD_x_rand, *NGD_param_rand)
plt.figure()
plt.hist(imgx_flat, num_bin, normed=True, label='original histogram', lw=3, fc=(0, 0, 1, 0.6))
plt.hist(imgx_flat_rand, num_bin, normed=True, label='noise histogram', lw=3, fc=(1, 0, 0, 0.6))
plt.plot(NGD_x_rand, NGD_curve_rand, 'r--', lw=5, alpha=0.6, label='noise fitted norm pdf')
mean = np.sum(imgx_flat_rand) / imgx_flat_rand.shape[0]
var = np.var(imgx_flat_rand)
std = np.std(imgx_flat_rand)
kur = scipy.stats.kurtosis(imgx_flat_rand)
plt.xlabel('Bins of gradient differences')
plt.ylabel('Frequency')
title_str = ('Histogram of horizontal gradients: mean={0:.4f}, variance={1:.4f}, kurtosis={2:.4f}'
             .format(mean, var, kur))
plt.title(title_str)
plt.legend()
plt.grid(True)


plt.figure()
plt.hist(imgx_flat, num_bin, log=True, normed=True, label='original histogram', lw=3, fc=(0, 0, 1, 0.6))
plt.hist(imgx_flat_rand, num_bin, log=True, normed=True, label='noise histogram', lw=3, fc=(1, 0, 0, 0.6))
plt.plot(NGD_x_rand, NGD_curve_rand, 'r--', lw=5, alpha=0.6, label='noise fitted norm pdf')
plt.xlabel('Bins of gradient differences')
plt.ylabel('Log of frequency')
plt.title('Log histogram of horizontal gradients')
plt.legend()
plt.grid(True)


# After down-sampling
img_rand_down1 = scipy.misc.imresize(img_rand, (int(img_rand.shape[0]/2), int(img_rand.shape[1]/2)), 'bilinear')
img_rand_down1 = grayRescale(img_rand_down1)
imgx_rand_down1 = gradientHorizontal(img_rand_down1)


imgx_rand_down1_flat = imgx_rand_down1.flatten()
NGD_param_rand_down1 = scipy.stats.norm.fit(imgx_rand_down1_flat, floc=0)
NGD_x_rand_down1 = np.linspace(imgx_rand_down1_flat.min(), imgx_rand_down1_flat.max(), imgx_rand_down1_flat.shape[0])
NGD_curve_rand_down1 = scipy.stats.norm.pdf(NGD_x_rand_down1, *NGD_param_rand_down1)
plt.figure()
plt.hist(imgx_flat, num_bin, normed=True, label='original histogram', lw=3, fc=(0, 0, 1, 0.6))
plt.hist(imgx_rand_down1_flat, 30, normed=True, label='noise histogram (down-sampled)', lw=3, fc=(1, 0, 0, 0.6))
plt.plot(NGD_x_rand_down1, NGD_curve_rand_down1, 'r--', lw=5, alpha=0.6, label='noise (down-sampled) fitted norm pdf')
mean = np.sum(imgx_rand_down1_flat) / imgx_rand_down1_flat.shape[0]
var = np.var(imgx_rand_down1_flat)
std = np.std(imgx_rand_down1_flat)
kur = scipy.stats.kurtosis(imgx_rand_down1_flat)
plt.xlabel('Bins of gradient differences')
plt.ylabel('Frequency')
title_str = ('Histogram of horizontal gradients: mean={0:.4f}, variance={1:.4f}, kurtosis={2:.4f}'
             .format(mean, var, kur))
plt.title(title_str)
plt.legend()
plt.grid(True)


plt.figure()
plt.hist(imgx_flat, num_bin, log=True, normed=True, label='original histogram', lw=3, fc=(0, 0, 1, 0.6))
plt.hist(imgx_rand_down1_flat, 30, log=True, normed=True, label='noise histogram (down-sampled)', lw=3, fc=(1, 0, 0, 0.6))
plt.plot(NGD_x_rand_down1, NGD_curve_rand_down1, 'r--', lw=5, alpha=0.6, label='noise (down-sampled) fitted norm pdf')
plt.xlabel('Bins of gradient differences')
plt.ylabel('Log of frequency')
plt.title('Log histogram of horizontal gradients')
plt.legend()
plt.grid(True)