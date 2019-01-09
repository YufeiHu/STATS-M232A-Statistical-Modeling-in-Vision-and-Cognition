# -*- coding: utf-8 -*-
import matplotlib as mplt
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line_aa


font = {'weight':'bold', 'size':12}
mplt.rc('font', **font)


def downsampling(lines_x, lines_y, lines_ori, lines_len):
    lines_x_down = lines_x / 2.0
    lines_y_down = lines_y / 2.0
    lines_ori_down = np.copy(lines_ori)
    lines_len_down = lines_len / 2.0
    
    lines_x_down = lines_x_down.astype(int)
    lines_y_down = lines_y_down.astype(int)
    
    index_del = []
    for i in range(lines_x_down.shape[0]):
        if lines_len_down[i] < 1:
            index_del.append(i)
    
    lines_x_down = np.delete(lines_x_down, index_del)
    lines_y_down = np.delete(lines_y_down, index_del)
    lines_ori_down = np.delete(lines_ori_down, index_del)
    lines_len_down = np.delete(lines_len_down, index_del)
    
    return lines_x_down, lines_y_down, lines_ori_down, lines_len_down
    
    
def drawLine(img, xCentre, yCentre, orientation, length):
    width = img.shape[0]
    height = img.shape[1]
    
    halfLen = length / 2
    cosOrient = np.cos(orientation)
    sinOrient = np.sin(orientation)
    xcoords = xCentre + halfLen * np.array([cosOrient, -cosOrient])
    ycoords = yCentre + halfLen * np.array([-sinOrient, sinOrient])
    xcoords = xcoords.astype(int)
    ycoords = ycoords.astype(int)
    
    if xcoords[0] < 0:
        xcoords[0] = 0
    elif xcoords[0] >= width:
        xcoords[0] = width - 1
        
    if xcoords[1] < 0:
        xcoords[1] = 0
    elif xcoords[1] >= width:
        xcoords[1] = width - 1
    
    if ycoords[0] < 0:
        ycoords[0] = 0
    elif ycoords[0] >= height:
        ycoords[0] = height - 1
        
    if ycoords[1] < 0:
        ycoords[1] = 0
    elif ycoords[1] >= height:
        ycoords[1] = height - 1
    
    r, c, _ = line_aa(xcoords[0], ycoords[0], xcoords[1], ycoords[1])
    img[r, c] = 0


def generateLength(cpf, len_pool):
    P = np.random.rand()
    for i in range(cpf.shape[0] - 1):
        if cpf[i+1] >= P:
            break
    length = len_pool[i]
    return length


N = 10000
width = 1024
height = 1024


if N > width * height:
    raise ValueError('Error: N is too big')


lines_x = np.random.randint(width, size=N)
lines_y = np.random.randint(height, size=N)
lines_ori = np.random.rand(N) * np.pi * 2
lines_len = np.zeros(N)


len_pool = np.linspace(1, 100, N)
pr = 1 / np.power(len_pool, 3)
pr = pr / np.sum(pr)
cpf = np.zeros(pr.shape[0])
for i in range(N):
    cpf[i] = np.sum(pr[0:i])


img = np.ones((width, height))
for i in range(N):
    length = generateLength(cpf, len_pool)
    while length < 1:
        length = generateLength(cpf, len_pool)
    lines_len[i] = length
    drawLine(img, lines_x[i], lines_y[i], lines_ori[i], lines_len[i])


# Question (1)
plt.figure()
plt.imshow(img, cmap = plt.get_cmap('gray'))
plt.title('Random Image')
plt.xticks([])
plt.yticks([])


# Question (2)
lines_x_down1, lines_y_down1, lines_ori_down1, lines_len_down1 = downsampling(lines_x, lines_y, lines_ori, lines_len)
lines_x_down2, lines_y_down2, lines_ori_down2, lines_len_down2 = downsampling(lines_x_down1, lines_y_down1, lines_ori_down1, lines_len_down1)


img_down1 = np.ones((int(width/2), int(height/2)))
for i in range(lines_x_down1.shape[0]):
    drawLine(img_down1, lines_x_down1[i], lines_y_down1[i], lines_ori_down1[i], lines_len_down1[i])


img_down2 = np.ones((int(width/4), int(height/4)))
for i in range(lines_x_down2.shape[0]):
    drawLine(img_down2, lines_x_down2[i], lines_y_down2[i], lines_ori_down2[i], lines_len_down2[i])


plt.figure()
plt.subplot(121)
plt.imshow(img_down1, cmap = plt.get_cmap('gray'))
plt.title('Random Image (down-sampled 1 time)')
plt.xticks([])
plt.yticks([])
plt.subplot(122)
plt.imshow(img_down2, cmap = plt.get_cmap('gray'))
plt.title('Random Image (down-sampled 2 time)')
plt.xticks([])
plt.yticks([])


# Question (3)
plt.figure()
for i in range(2):
    plt.subplot(3, 2, i+1)
    pos_x = np.random.randint(width - 128)
    pos_y = np.random.randint(height - 128)
    plt.imshow(img[pos_x:pos_x+128, pos_y:pos_y+128], cmap = plt.get_cmap('gray'))
    plt.title('Patch of random image')
    plt.xticks([])
    plt.yticks([])
for i in range(2):
    plt.subplot(3, 2, i+3)
    pos_x = np.random.randint(width/2 - 128)
    pos_y = np.random.randint(height/2 - 128)
    plt.imshow(img_down1[pos_x:pos_x+128, pos_y:pos_y+128], cmap = plt.get_cmap('gray'))
    plt.title('Patch of random image (down-sampled 1 time)')
    plt.xticks([])
    plt.yticks([])
for i in range(2):
    plt.subplot(3, 2, i+5)
    pos_x = np.random.randint(width/4 - 128)
    pos_y = np.random.randint(height/4 - 128)
    plt.imshow(img_down2[pos_x:pos_x+128, pos_y:pos_y+128], cmap = plt.get_cmap('gray'))
    plt.title('Patch of random image (down-sampled 2 time)')
    plt.xticks([])
    plt.yticks([])