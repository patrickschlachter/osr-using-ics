#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
# @Time     : 2018/11/25 10:15
# @Author   : Yiwen Liao
# @File     : visualizations.py 
# @Software : PyCharm
# @Contact  : yiwen.liao93@gmail.com


import math
import itertools
import time as t
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from umap import UMAP


def img_visualize(data, interpolation=None, num_to_show=None, shuffle_img=False, to_save=False, name=''):
    """Visualize a batch of images with desired amount.
    Visualize a batch of images. Used as internal functions of other visualization functions.
    Input should only have 3 dimensions with (samples, width, height).

    :param data: Numpy array for a batch of images.
    :param interpolation: Boolean. If use interpolation for kernels.
    :param num_to_show: The number of images to show.
    :param shuffle_img:Boolean. If randomly select the images to be shown.
    :param to_save: Default False. If True, then all the plots will be saved.
    :param name: File name for saving plots.
    :return: None
    """

    fig_width = 10
    fig_height = 10

    num_sample = data.shape[0]
    n = min(num_to_show, int(np.sqrt(num_sample)))

    digit_size = data.shape[1]
    if len(data.shape) > 3 and data.shape[-1] == 3:
        figure = np.zeros((digit_size * n, digit_size * n, 3))
    else:
        figure = np.zeros((digit_size * n, digit_size * n))

    img_order = np.arange(0, n * n)

    if shuffle_img:
        np.random.shuffle(img_order)

    for i in range(n):
        for j in range(n):
            index = img_order[n * i + j]
            digit = data[index, ...]
            if len(digit.shape) > 2 and digit.shape[-1] == 1:
                digit = np.squeeze(digit, axis=-1)
            if len(data.shape) > 3 and data.shape[-1] == 3:
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size, :] = digit
            else:
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(fig_width, fig_height))
    if len(data.shape) > 3 and data.shape[-1] == 3:
        plt.imshow(figure, interpolation=interpolation)
    else:
        plt.imshow(figure, cmap='Greys_r', interpolation=interpolation)

    plt.axis('off')

    if to_save:
        plt.savefig(name, dpi=500)
        plt.close()
    else:
        plt.show()

