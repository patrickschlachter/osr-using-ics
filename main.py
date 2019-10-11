#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
# @Time     : 2018/11/20 10:09
# @Author   : Yiwen Liao
# @File     : main.py 
# @Software : PyCharm
# @Contact  : yiwen.liao93@gmail.com


from models.osrnet import *


def run_osrnet(normal_class=None, dataset=None):
    """Run OSRNET for open set recognition

    :param normal_class: A list of class labels that are considered as known classes during training.
    :param dataset: The name of a desired dataset: mnist, fmnist or cifar10.
    """

    set_seed()

    # ========== constants ==========
    RHO = 10
    LOSS_WEIGHTS = [1., 1.]
    SPLIT_METHOD = 'cnn'
    TRAIN_EPOCH = 100
    PRE_TRAIN_EPOCH = 35

    data = get_data(dataset,
                    normal_class=normal_class,
                    data_format='tensor')

    num_cls = len(normal_class)
    name = dataset + '_'
    for idx in range(num_cls):
        name = name + str(normal_class[idx])

    # Train a network for splitting the known classes
    train_logits_cnn(data=data['x_train_normal'],
                     label=data['y_train_normal'],
                     normal_class=normal_class,
                     reg=1e-3,
                     epoch=100,
                     batch_size=128,
                     name=name)

    # Train an OSRNET
    train_joint_osrnet(data=data,
                       name=name,
                       rho=RHO,
                       reg=1e-3,
                       latent_fea=256,
                       num_epoch=TRAIN_EPOCH,
                       batch_size=64,
                       split_method=SPLIT_METHOD,
                       normal_class=normal_class,
                       backbone='modified_vgg',
                       loss_weights=LOSS_WEIGHTS,
                       pretrain_ep=PRE_TRAIN_EPOCH)


if __name__ == '__main__':
    run_osrnet(normal_class=[1, 3, 4, 6, 7, 9], dataset='cifar10')
