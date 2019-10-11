#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
# @Time     : 2018/12/02 22:42
# @Author   : Yiwen Liao
# @File     : cnn_pool.py 
# @Software : PyCharm
# @Contact  : yiwen.liao93@gmail.com


from packages import *
from toolkits.utils import assign_label
from keras.applications import vgg16, vgg19, xception, densenet, resnet50, inception_v3


def build_cnn(img_height=None, num_channel=None, reg=None, latent_fea=None, num_normal_class=None, cnn_type=None):
    """Build CNN or OSRNET.

    :param img_height: Input image height (width should be equal to this).
    :param num_channel: Number of image channels.
    :param reg: Decay of regularization terms.
    :param latent_fea: Number of latent features.
    :param num_normal_class: Number of known classes.
    :param cnn_type: The name of the CNN used as a backbone: modified_vgg, alexnet, mlp, densenet, etc.
    :return: A list of models.
    """

    # ==================== Constants Definition ====================
    acti_func = 'linear'
    clf_acti = 'softmax'

    acti_alpha = 0.2
    set_bias = False

    weights_init = tn(mean=0, stddev=0.01)
    # weights_init = 'glorot_uniform'

    bn_eps = 1e-3
    bn_m = 0.99

    logits_layer = None

    # ==================== General Input Layer ====================
    input_layer = Input(shape=(img_height, img_height, num_channel), name='input_layer')

    # ==================== CNN Pool ====================
    if cnn_type == 'alexnet':

        # ==================== AlexNet ====================

        conv_1 = Conv2D(filters=96, kernel_size=(5, 5), activation=acti_func, name='conv_1',
                        dilation_rate=(4, 4), kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                        kernel_initializer=weights_init)(input_layer)

        lrelu_1 = LeakyReLU(alpha=acti_alpha)(conv_1)

        pool_1 = MaxPooling2D(pool_size=(2, 2), name='pool_1')(lrelu_1)

        bn_1 = BatchNormalization(name='bn_1')(pool_1)

        conv_2 = Conv2D(filters=256, kernel_size=(3, 3), activation=acti_func, name='conv_2',
                        dilation_rate=(2, 2), padding='same',
                        kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                        kernel_initializer=weights_init)(bn_1)

        lrelu_2 = LeakyReLU(alpha=acti_alpha)(conv_2)

        pool_2 = MaxPooling2D(pool_size=(2, 2), name='pool_2')(lrelu_2)

        bn_2 = BatchNormalization(name='bn_2')(pool_2)

        conv_3 = Conv2D(filters=384, kernel_size=(3, 3), activation=acti_func, name='conv_3',
                        padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                        kernel_initializer=weights_init)(bn_2)

        lrelu_3 = LeakyReLU(alpha=acti_alpha)(conv_3)

        bn_3 = BatchNormalization(name='bn_3')(lrelu_3)

        conv_4 = Conv2D(filters=384, kernel_size=(3, 3), activation=acti_func, name='conv_4',
                        padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                        kernel_initializer=weights_init)(bn_3)

        lrelu_4 = LeakyReLU(alpha=acti_alpha)(conv_4)

        bn_4 = BatchNormalization(name='bn_4')(lrelu_4)

        conv_5 = Conv2D(filters=256, kernel_size=(3, 3), activation=acti_func, name='conv_5',
                        padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                        kernel_initializer=weights_init)(bn_4)

        lrelu_5 = LeakyReLU(alpha=acti_alpha)(conv_5)

        # pool_6 = MaxPooling2D(pool_size=(2, 2), name='pool_6')(lrelu_5)
        #
        # flt_7 = Flatten(name='flt_7')(pool_6)

        flt_7 = GlobalAveragePooling2D()(lrelu_5)

        dense_8 = Dense(units=4096, activation=acti_func, name='dense_8',
                        kernel_regularizer=regularizers.l2(reg), use_bias=not set_bias,
                        kernel_initializer=weights_init)(flt_7)

        lrelu_8 = LeakyReLU(alpha=acti_alpha)(dense_8)

        drop_8 = Dropout(rate=0.5)(lrelu_8)

        dense_9 = Dense(units=latent_fea, activation=acti_func, name='dense_9',
                        kernel_regularizer=regularizers.l2(reg), use_bias=not set_bias,
                        kernel_initializer=weights_init)(drop_8)

        lrelu_9 = LeakyReLU(alpha=acti_alpha)(dense_9)

        drop_9 = Dropout(rate=0.5)(lrelu_9)

        dense_10 = Dense(units=num_normal_class, activation='softmax', name='dense_10',
                         kernel_regularizer=regularizers.l2(reg), use_bias=not set_bias,
                         kernel_initializer=weights_init)(drop_9)

        # dense_10 = RBFLayer(output_dim=num_normal_class)(drop_9)
        top_layer = Reshape(target_shape=(-1,), name='top_layer')(dense_10)
        latent_layer = Reshape(target_shape=(-1,), name='latent_layer')(dense_9)
        # latent_layer = Reshape(target_shape=(-1,), name='latent_layer')(lrelu_9)

    elif cnn_type == 'modified_vgg':

        conv_1 = Conv2D(filters=32, kernel_size=(3, 3), activation=acti_func, name='conv_1',
                        padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                        kernel_initializer=weights_init)(input_layer)
        conv_11 = Conv2D(filters=32, kernel_size=(3, 3), activation=acti_func, name='conv_11',
                         padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                         kernel_initializer=weights_init)(conv_1)
        conv_1 = Concatenate()([conv_1, conv_11])  # 32x32x64

        lrelu_1 = LeakyReLU(alpha=acti_alpha)(conv_1)

        pool_1 = AveragePooling2D(pool_size=(2, 2), name='pool_1')(lrelu_1)  # 16x16 / 14x14

        bn_1 = BatchNormalization(momentum=bn_m, epsilon=bn_eps, name='bn_1')(pool_1)

        conv_2 = Conv2D(filters=64, kernel_size=(3, 3), activation=acti_func, name='conv_2',
                        padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                        kernel_initializer=weights_init)(bn_1)
        conv_22 = Conv2D(filters=64, kernel_size=(3, 3), activation=acti_func, name='conv_22',
                         padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                         kernel_initializer=weights_init)(conv_2)
        conv_2 = Concatenate()([conv_2, conv_22])  # 16x16x128

        lrelu_2 = LeakyReLU(alpha=acti_alpha)(conv_2)

        pool_2 = AveragePooling2D(pool_size=(2, 2), name='pool_2')(lrelu_2)  # 8x8 / 7x7

        if img_height == 28:
            pool_2 = ZeroPadding2D(padding=(1, 1))(pool_2)  # zero-padding if mnist or fashion-mnist

        bn_2 = BatchNormalization(momentum=bn_m, epsilon=bn_eps, name='bn_2')(pool_2)

        conv_3 = Conv2D(filters=128, kernel_size=(3, 3), activation=acti_func, name='conv_3',
                        padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                        kernel_initializer=weights_init)(bn_2)
        conv_33 = Conv2D(filters=128, kernel_size=(3, 3), activation=acti_func, name='conv_33',
                         padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                         kernel_initializer=weights_init)(conv_3)
        conv_3 = Concatenate()([conv_3, conv_33])  # 8x8x256

        lrelu_3 = LeakyReLU(alpha=acti_alpha)(conv_3)

        pool_3 = AveragePooling2D(pool_size=(2, 2), name='pool_3')(lrelu_3)  # 4x4

        bn_3 = BatchNormalization(momentum=bn_m, epsilon=bn_eps, name='bn_3')(pool_3)

        conv_4 = Conv2D(filters=256, kernel_size=(3, 3), activation=acti_func, name='conv_4',
                        padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                        kernel_initializer=weights_init)(bn_3)
        conv_44 = Conv2D(filters=256, kernel_size=(3, 3), activation=acti_func, name='conv_44',
                         padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                         kernel_initializer=weights_init)(conv_4)
        conv_4 = Concatenate()([conv_4, conv_44])  # 4x4x512

        lrelu_4 = LeakyReLU(alpha=acti_alpha)(conv_4)

        pool_4 = AveragePooling2D(pool_size=(2, 2), name='pool_4')(lrelu_4)  # 2x2

        bn_4 = BatchNormalization(momentum=bn_m, epsilon=bn_eps, name='bn_4')(pool_4)

        conv_5 = Conv2D(filters=256, kernel_size=(1, 1), activation=acti_func, name='conv_5',
                        kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                        kernel_initializer=weights_init)(bn_4)  # 2x2
        conv_55 = Conv2D(filters=256, kernel_size=(1, 1), activation=acti_func, name='conv_55',
                         kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                         kernel_initializer=weights_init)(bn_4)
        conv_5 = Concatenate()([conv_5, conv_55])  # 2x2x512

        lrelu_5 = LeakyReLU(alpha=acti_alpha)(conv_5)

        bn_5 = BatchNormalization(name='bn_5')(lrelu_5)

        # flt_7 = GlobalAveragePooling2D()(bn_5)
        flt_7 = Flatten()(bn_5)

        dense_8 = Dense(units=256, activation=acti_func, name='dense_8',
                        kernel_regularizer=regularizers.l2(reg), use_bias=not set_bias,
                        kernel_initializer=weights_init)(flt_7)

        lrelu_8 = LeakyReLU(alpha=acti_alpha)(dense_8)

        drop_8 = Dropout(rate=0.5)(lrelu_8)
        # drop_8 = BatchNormalization()(lrelu_8)

        dense_9 = Dense(units=latent_fea, activation=acti_func, name='dense_9',
                        kernel_regularizer=regularizers.l2(reg), use_bias=not set_bias,
                        kernel_initializer=weights_init)(drop_8)

        lrelu_9 = LeakyReLU(alpha=acti_alpha)(dense_9)

        drop_9 = Dropout(rate=0.5)(lrelu_9)
        # drop_9 = BatchNormalization()(lrelu_9)

        dense_10 = Dense(units=num_normal_class, activation='linear', name='dense_10',
                         kernel_regularizer=regularizers.l2(reg), use_bias=not set_bias,
                         kernel_initializer=weights_init)(drop_9)
        sf_10 = Softmax()(dense_10)

        top_layer = Reshape(target_shape=(-1,), name='top_layer')(sf_10)
        latent_layer = Reshape(target_shape=(-1,), name='latent_layer')(lrelu_9)
        logits_layer = Reshape(target_shape=(-1,), name='logits_layer')(dense_10)

    elif cnn_type == 'logits_cnn':

        conv_1 = Conv2D(filters=32, kernel_size=(7, 7), activation=acti_func, name='conv_1',
                        padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                        kernel_initializer=weights_init)(input_layer)
        conv_11 = Conv2D(filters=32, kernel_size=(7, 7), activation=acti_func, name='conv_11',
                         padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                         kernel_initializer=weights_init)(conv_1)
        conv_1 = Concatenate()([conv_1, conv_11])

        lrelu_1 = LeakyReLU(alpha=acti_alpha)(conv_1)

        pool_1 = AveragePooling2D(pool_size=(2, 2), name='pool_1')(lrelu_1)  # 16x16 / 14x14

        bn_1 = BatchNormalization(name='bn_1')(pool_1)

        conv_2 = Conv2D(filters=64, kernel_size=(3, 3), activation=acti_func, name='conv_2',
                        padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                        kernel_initializer=weights_init)(bn_1)
        conv_22 = Conv2D(filters=64, kernel_size=(3, 3), activation=acti_func, name='conv_22',
                         padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                         kernel_initializer=weights_init)(conv_2)
        conv_2 = Concatenate()([conv_2, conv_22])

        lrelu_2 = LeakyReLU(alpha=acti_alpha)(conv_2)

        pool_2 = AveragePooling2D(pool_size=(2, 2), name='pool_2')(lrelu_2)  # 8x8 / 7x7

        if img_height == 28:
            pool_2 = ZeroPadding2D(padding=(1, 1))(pool_2)

        bn_2 = BatchNormalization(name='bn_2')(pool_2)

        conv_3 = Conv2D(filters=128, kernel_size=(3, 3), activation=acti_func, name='conv_3',
                        padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                        kernel_initializer=weights_init)(bn_2)
        conv_33 = Conv2D(filters=128, kernel_size=(3, 3), activation=acti_func, name='conv_33',
                         padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                         kernel_initializer=weights_init)(conv_3)
        conv_3 = Concatenate()([conv_3, conv_33])

        lrelu_3 = LeakyReLU(alpha=acti_alpha)(conv_3)

        pool_3 = AveragePooling2D(pool_size=(2, 2), name='pool_3')(lrelu_3)  # 4x4

        bn_3 = BatchNormalization(name='bn_3')(pool_3)

        conv_4 = Conv2D(filters=256, kernel_size=(3, 3), activation=acti_func, name='conv_4',
                        padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                        kernel_initializer=weights_init)(bn_3)
        conv_44 = Conv2D(filters=256, kernel_size=(3, 3), activation=acti_func, name='conv_44',
                         padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                         kernel_initializer=weights_init)(conv_4)
        conv_4 = Concatenate()([conv_4, conv_44])

        lrelu_4 = LeakyReLU(alpha=acti_alpha)(conv_4)

        pool_4 = AveragePooling2D(pool_size=(2, 2), name='pool_4')(lrelu_4)  # 2x2

        bn_4 = BatchNormalization(name='bn_4')(pool_4)

        conv_5 = Conv2D(filters=256, kernel_size=(1, 1), activation=acti_func, name='conv_5',
                        kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                        kernel_initializer=weights_init)(bn_4)  # 2x2
        conv_55 = Conv2D(filters=256, kernel_size=(1, 1), activation=acti_func, name='conv_55',
                         kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                         kernel_initializer=weights_init)(bn_4)
        conv_5 = Concatenate()([conv_5, conv_55])

        lrelu_5 = LeakyReLU(alpha=acti_alpha)(conv_5)

        bn_5 = BatchNormalization(name='bn_5')(lrelu_5)

        # flt_7 = GlobalAveragePooling2D()(bn_5)
        flt_7 = Flatten()(bn_5)

        dense_8 = Dense(units=256, activation=acti_func, name='dense_8',
                        kernel_regularizer=regularizers.l2(reg), use_bias=not set_bias,
                        kernel_initializer=weights_init)(flt_7)

        lrelu_8 = LeakyReLU(alpha=acti_alpha)(dense_8)

        drop_8 = Dropout(rate=0.5)(lrelu_8)
        # drop_8 = BatchNormalization()(lrelu_8)

        dense_9 = Dense(units=latent_fea, activation=acti_func, name='dense_9',
                        kernel_regularizer=regularizers.l2(reg), use_bias=not set_bias,
                        kernel_initializer=weights_init)(drop_8)

        lrelu_9 = LeakyReLU(alpha=acti_alpha)(dense_9)

        drop_9 = Dropout(rate=0.5)(lrelu_9)
        # drop_9 = BatchNormalization()(lrelu_9)

        dense_10 = Dense(units=num_normal_class, activation='linear', name='dense_10',
                         kernel_regularizer=regularizers.l2(reg), use_bias=not set_bias,
                         kernel_initializer=weights_init)(drop_9)
        clf_layer = Softmax(name='softmax')(dense_10)

        top_layer = Reshape(target_shape=(-1,), name='top_layer')(clf_layer)
        latent_layer = Reshape(target_shape=(-1,), name='latent_layer')(dense_10)
        logits_layer = Reshape(target_shape=(-1,), name='latent_layer')(dense_10)

    elif cnn_type == 'mlp':
        conv_1 = Conv2D(filters=256, kernel_size=(5, 5), activation=acti_func, name='conv_1',
                        dilation_rate=(2, 2), kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                        kernel_initializer=weights_init)(input_layer)

        lrelu_1 = LeakyReLU(alpha=acti_alpha)(conv_1)

        bn_1 = BatchNormalization(name='bn_1')(lrelu_1)

        conv_2 = Conv2D(filters=latent_fea, kernel_size=(5, 5), activation=acti_func, name='conv_2',
                        dilation_rate=(2, 2), kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                        kernel_initializer=weights_init)(bn_1)

        lrelu_2 = LeakyReLU(alpha=acti_alpha)(conv_2)
        latent_layer = GlobalAveragePooling2D()(lrelu_2)
        top_layer = Dense(units=num_normal_class, activation='softmax', name='dense_10',
                          kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                          kernel_initializer=weights_init)(latent_layer)

    elif cnn_type == 'vgg16':
        if img_height < 48:
            img_size = 2 * img_height
            ups = UpSampling2D((2, 2))(input_layer)
        else:
            ups = input_layer
            img_size = img_height
        if num_channel != 3:
            conc = Concatenate()([ups, ups, ups])
        else:
            conc = ups
        latent_layer = vgg16.VGG16(include_top=False, weights=None,
                                   input_shape=(img_size, img_size, 3),
                                   pooling='avg')(conc)
        logits_layer = Dense(units=num_normal_class, activation='linear', name='logits_layer',
                             kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                             kernel_initializer=weights_init)(latent_layer)
        top_layer = Softmax()(logits_layer)

    elif cnn_type == 'vgg19':
        if img_height < 48:
            img_size = 2 * img_height
            ups = UpSampling2D((2, 2))(input_layer)
        else:
            ups = input_layer
            img_size = img_height
        if num_channel != 3:
            conc = Concatenate()([ups, ups, ups])
        else:
            conc = ups
        latent_layer = vgg19.VGG19(include_top=False, weights=None,
                                   input_shape=(img_size, img_size, 3),
                                   pooling='avg')(conc)
        top_layer = Dense(units=num_normal_class, activation='softmax',
                          kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                          kernel_initializer=weights_init)(latent_layer)

    elif cnn_type == 'densenet':
        if img_height == 28:
            zero_padding = ZeroPadding2D((2, 2))(input_layer)
            conc = Concatenate()([zero_padding, zero_padding, zero_padding])
        else:
            conc = input_layer

        latent_layer = densenet.DenseNet121(include_top=False, weights=None,
                                            input_shape=(32, 32, 3),
                                            pooling='avg')(conc)
        top_layer = Dense(units=num_normal_class, activation='softmax',
                          kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                          kernel_initializer=weights_init)(latent_layer)

    elif cnn_type == 'xception':
        img_size = 3 * img_height
        ups = UpSampling2D((3, 3))(input_layer)
        if num_channel != 3:
            conc = Concatenate()([ups, ups, ups])
        else:
            conc = ups
        latent_layer = xception.Xception(include_top=False, weights=None,
                                         input_shape=(img_size, img_size, 3),
                                         pooling='avg')(conc)
        top_layer = Dense(units=num_normal_class, activation='softmax',
                          kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                          kernel_initializer=weights_init)(latent_layer)

    elif cnn_type == 'resnet':
        if img_height == 28:
            ups = UpSampling2D((8, 8))(input_layer)
            conc = Concatenate()([ups, ups, ups])
        else:
            ups = UpSampling2D((7, 7))(input_layer)
            conc = ups
        latent_layer = resnet50.ResNet50(include_top=False, weights=None,
                                         input_shape=(224, 224, 3),
                                         pooling='avg')(conc)
        top_layer = Dense(units=num_normal_class, activation='softmax',
                          kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                          kernel_initializer=weights_init)(latent_layer)

    elif cnn_type == 'inception':
        img_size = 5 * img_height
        ups = UpSampling2D((5, 5))(input_layer)
        if num_channel != 3:
            conc = Concatenate()([ups, ups, ups])
        else:
            conc = ups
        latent_layer = inception_v3.InceptionV3(include_top=False, weights=None,
                                                input_shape=(img_size, img_size, 3),
                                                pooling='avg')(conc)
        top_layer = Dense(units=num_normal_class, activation='softmax',
                          kernel_regularizer=regularizers.l2(reg), use_bias=set_bias,
                          kernel_initializer=weights_init)(latent_layer)

    else:
        raise ValueError('No suitable CNN architecture...')

    cnn = Model(inputs=input_layer, outputs=top_layer, name=cnn_type)
    cnn_latent = Model(inputs=input_layer, outputs=latent_layer, name=cnn_type + '_latent')
    cnn_logits = Model(inputs=input_layer, outputs=logits_layer, name=cnn_type + '_logits')

    # ==================== Intra-Class Networks ====================

    input_1 = Input(shape=(img_height, img_height, num_channel), name='input_1')
    input_2 = Input(shape=(img_height, img_height, num_channel), name='input_2')

    lat_1 = cnn_latent(input_1)
    lat_2 = cnn_latent(input_2)

    latent_dist = Subtract(name='latent_dist')([lat_1, lat_2])

    dense_ly = Dense(units=1, activation='sigmoid', name='dense_ly',
                     kernel_regularizer=regularizers.l2(reg))(latent_dist)

    ic_network = Model(inputs=[input_1, input_2], outputs=dense_ly)

    # ==================== Joint layers ====================
    dense_11 = Dense(units=num_normal_class-1, activation='softmax', name='dense_joint',
                     kernel_regularizer=regularizers.l2(reg), use_bias=not set_bias,
                     kernel_initializer=weights_init)(latent_layer)
    joint_layer = Reshape(target_shape=(-1,), name='joint_layer')(dense_11)

    joint_cnn = Model(inputs=input_layer, outputs=[top_layer, joint_layer])

    cnn.summary()
    ic_network.summary()

    return cnn, cnn_latent, joint_cnn, cnn_logits


def train_logits_cnn(data=None, label=None, normal_class=None, reg=None, epoch=None, batch_size=None, name=None):
    """Train a normal CNN and save the layers from to bottom layer to logit output layer for further data splitting.

    :param data: Training data in a 4D tensor.
    :param label: Corresponding original labels for the training data.
    :param normal_class: A list of selected known classes labels.
    :param reg: Decay for the regularization term.
    :param epoch: Trianing epochs.
    :param batch_size: The size of batch sizes.
    :param name: The name of the CNN for saving.
    :return: CNN models.
    """

    num_normal_class = len(normal_class)
    label = to_categorical(assign_label(normal_class=normal_class, original_label=label, include_zero=True))
    num_img, img_height, img_width, num_channel = data.shape[0], data.shape[1], data.shape[2], data.shape[-1]

    model_set = build_cnn(img_height=img_height, num_channel=num_channel, reg=reg,
                          latent_fea=256, num_normal_class=num_normal_class, cnn_type='logits_cnn')

    cnn = model_set[0]
    cnn_latent = model_set[1]

    customized_optimizer = optimizers.rmsprop(lr=1e-4, decay=1e-9)
    cnn.compile(optimizer=customized_optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    cnn_latent.compile(optimizer=customized_optimizer,
                       loss='mse')

    cnn.fit(x=data,
            y=label,
            batch_size=batch_size,
            epochs=epoch,
            verbose=2)

    if name is not None:
        cnn_latent.save('./trained_models/cnn_for_ds_' + name + '.h5')
        return
    else:
        return cnn, cnn_latent

