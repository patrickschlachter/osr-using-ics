#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
# @Time     : 2018/12/02 22:59
# @Author   : Yiwen Liao
# @File     : osrnet.py 
# @Software : PyCharm
# @Contact  : yiwen.liao93@gmail.com


from .cnn_pool import *
from toolkits.utils import *
from toolkits.evaluations import *
from toolkits.visualizations import *


def train_joint_osrnet(data=None, name=None, rho=None, reg=None, latent_fea=None, num_epoch=None, batch_size=64,
                       split_method='cnn', normal_class=None, backbone='modified_vgg',
                       loss_weights=None, pretrain_ep=None):

    # ==================== split training data ====================
    typical_index, atypical_index = split_data(model_name=name,
                                               data=data['x_train_normal'],
                                               rho=rho,
                                               split_method=split_method,
                                               ground_truth=data['y_train_normal'],
                                               normal_class=normal_class)

    # ==================== assign labels to typical and atypical data====================
    typical_label = data['y_train_normal'][typical_index]
    atypical_label = data['y_train_normal'][atypical_index] * 0

    typical_label = assign_label(normal_class=normal_class, original_label=typical_label, include_zero=False)
    print('\nThere are %d typical normal samples and %d atypical normal samples...' % (len(typical_label),
                                                                                       len(atypical_label)))

    # assign labels for closed-set regularization
    normal_label = assign_label(normal_class=normal_class, original_label=data['y_train_normal'], include_zero=True)
    normal_lb_ty = normal_label[typical_index]
    normal_lb_aty = normal_label[atypical_index]
    normal_label = np.concatenate([normal_lb_ty, normal_lb_aty])

    # ==================== shuffle the training data ====================
    normal_x = np.vstack([data['x_train_normal'][typical_index], data['x_train_normal'][atypical_index]])
    normal_y = np.concatenate([typical_label, atypical_label])

    training_idx = np.random.permutation(np.arange(0, len(normal_y)))
    normal_x = normal_x[training_idx]
    normal_y = normal_y[training_idx]
    normal_label = normal_label[training_idx]

    # ==================== create and compile network ====================
    img_height, img_width = normal_x.shape[1], normal_x.shape[2]
    num_ch = normal_x.shape[-1]
    num_train = normal_x.shape[0]
    num_all_cls = 1 + len(normal_class)
    idx = np.arange(0, num_train)

    model_set = build_cnn(img_height=img_height, num_channel=num_ch, reg=reg, latent_fea=latent_fea,
                          num_normal_class=num_all_cls, cnn_type=backbone)

    customized_optimizer = optimizers.adam(lr=3e-4, beta_1=0.5, clipvalue=1.0, decay=1e-10)

    osrnet = model_set[0]
    osrnet_latent = model_set[1]
    osrnet_joint = model_set[2]
    osrnet_logits = model_set[3]

    osrnet.compile(optimizer=customized_optimizer,
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
    osrnet_latent.compile(optimizer=customized_optimizer, loss='mse')
    osrnet_joint.compile(optimizer=customized_optimizer,
                         loss=['categorical_crossentropy', 'categorical_crossentropy'],
                         loss_weights=loss_weights)
    osrnet_logits.compile(optimizer=customized_optimizer, loss='mse')

    # ==================== prepare test set ====================
    x_test = np.vstack([data['x_test_normal'], data['x_test_abnormal']])
    y_test_normal = assign_label(normal_class=normal_class, original_label=data['y_test_normal'], include_zero=False)
    y_test_gt = np.concatenate([y_test_normal, data['y_test_abnormal']*0])

    # ==================== train the network====================
    best_baccu = 0
    best_auc = 0
    best_cs_accu = 0

    res_baccu = []
    res_auc = []
    res_cs_accu = []
    res_train_accu = []

    record_step = 5

    # pre-train the network
    np.random.shuffle(idx)
    if pretrain_ep is not None:
        osrnet.fit(x=normal_x[idx],
                   y=to_categorical(normal_label+1)[idx],
                   batch_size=64,
                   epochs=pretrain_ep,
                   verbose=2)

    name = name + '_l1_%s_l2_%s' % (str(loss_weights[0]), str(loss_weights[1]))
    for i in range(num_epoch):

        if (i + 1) % record_step == 0 or i == 0:
            print('\nTraining for epoch %d' % (i+1))
            y_test_pred = np.argmax(osrnet.predict(x_test, batch_size=128), axis=-1)
            baccu = cal_openset_baccu(y_test_gt, y_test_pred, label_ref=np.arange(0, num_all_cls))

            y_test_decision_function = osrnet_logits.predict(x_test, batch_size=128)
            auc = cal_modified_auc(y_test_gt, y_test_decision_function)

            y_test_cs_pred = np.argmax(osrnet.predict(data['x_test_normal'], batch_size=128), axis=-1)
            cs_accu = cal_closed_set_accu(y_test_normal, y_test_cs_pred)

            y_train_pred = np.argmax(osrnet.predict(normal_x[:500], batch_size=128), axis=-1)
            train_cs_accu = cal_closed_set_accu(normal_label[:500]+1, y_train_pred)

            res_baccu.append(baccu)
            res_auc.append(auc)
            res_cs_accu.append(cs_accu)
            res_train_accu.append(train_cs_accu)

            if baccu > best_baccu:
                best_baccu = baccu
                best_auc = auc
                best_cs_accu = cs_accu
                osrnet.save(filepath='./trained_models/osrnet_best_%s_rho_%d.h5' % (name, rho))
                osrnet_logits.save(filepath='./trained_models/osrnet_logits_best_%s_rho_%d.h5' % (name, rho))

            print('\nThe best baccu is %.4f' % best_baccu)
            print('The best auc is %.4f' % best_auc)
            print('The best cs accu is %.4f' % best_cs_accu)

        osrnet_joint.fit(x=normal_x[idx],
                         y=[to_categorical(normal_y)[idx],
                            to_categorical(normal_label)[idx]],
                         batch_size=batch_size,
                         epochs=1,
                         verbose=0)
        np.random.shuffle(idx)

    osrnet.save(filepath='./trained_models/osrnet_end_%s_rho_%d.h5' % (name, rho))






