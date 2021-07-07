from PIL import Image, ImageEnhance
import numpy as np
from scipy.ndimage import filters

import os
def XDOG(im):
    """用来将图片生成简笔画"""
    Gamma = 0.9 #0.97  ##过滤线条
    Phi = 200
    Epsilon = 0.1
    k = 2.0 #2.5  ##过滤线条
    Sigma = 0.5 #1.5 ##调整细粒度

    im = im.convert('L')
    im = np.array(ImageEnhance.Sharpness(im).enhance(3.0))
    im2 = filters.gaussian_filter(im, Sigma)
    im3 = filters.gaussian_filter(im, Sigma* k)
    differencedIm2 = im2 - (Gamma * im3)
    (x, y) = np.shape(im2)
    for i in range(x):
        for j in range(y):
            if differencedIm2[i, j] < Epsilon:
                differencedIm2[i, j] = 1
            else:
                differencedIm2[i, j] = 250 + np.tanh(Phi * (differencedIm2[i, j]))
    gray_pic=differencedIm2.astype(np.uint8)
    final_img = Image.fromarray( gray_pic)
    return final_img
test_data="./test_img"
save="./sa"
test_imgs=os.listdir(test_data+"/data")
for i in test_imgs:
    img=Image.open(test_data+"/data/"+i)
    img=XDOG(img)
    img = img.convert('L')
    img=img.resize((256,256))
    img.save(save+"/data/"+i)
