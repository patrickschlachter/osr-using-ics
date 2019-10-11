#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
# @Time     : 2018/11/25 10:15
# @Author   : Yiwen Liao
# @File     : utils.py 
# @Software : PyCharm
# @Contact  : yiwen.liao93@gmail.com


import scipy.io
from packages import *


def set_seed(first_seed=2018):
    """Set seed for reproducible results.

    :param first_seed: Integer number as the global seed.
    """

    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(first_seed)
    rn.seed(10)

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(16)

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


def get_svhn():
    """Get SVHN dataset from directory.

    :return: Training data and test data in tensor.
    """

    file_path = './datasets/SVHN/'

    train_data = scipy.io.loadmat(file_path + 'train_32x32.mat')
    test_data = scipy.io.loadmat(file_path + 'test_32x32.mat')

    x_train = train_data['X']
    y_train = train_data['y']

    x_test = test_data['X']
    y_test = test_data['y']

    # Assign the digit zero with label of 0
    y_train_idx = np.where(y_train == 10)[0]
    y_train[y_train_idx] = 0
    y_test_idx = np.where(y_test == 10)[0]
    y_test[y_test_idx] = 0

    # Change channel first to channel last
    x_train = np.moveaxis(x_train, -1, 0)
    x_test = np.moveaxis(x_test, -1, 0)

    return x_train, y_train, x_test, y_test


def _extract_data(data=None, label=None, target_lb=None):
    """Extract dataset regarding given normal / abnormal labels-

    :param data: A numpy tensor. First axis should be the number of samples.
    :param label: The corresponding labels for the data.
    :param target_lb: An integer value standing for the only one known class.
    :return: normal data, abnormal data, normal labels, abnormal labels
    """

    index = 0
    if isinstance(target_lb, int):
        index = index + (label == target_lb) * 1
    else:
        for lb in target_lb:
            index = index + (label == lb) * 1

    normal_idx = np.where(index == 1)[0]
    abnormal_idx = np.where(index == 0)[0]

    data_normal = data[normal_idx]
    data_abnormal = data[abnormal_idx]

    label_normal = label[normal_idx]
    label_abnormal = label[abnormal_idx]

    return data_normal, data_abnormal, label_normal, label_abnormal


def _reshape_data(data=None, data_shape=None, num_channels=None):
    """Reshape image data into vectors / matrices / tensors.

    :param data: A numpy tensor. First axis should be the number of samples.
    :param data_shape: Desired data shape. It should be a string.
    :param num_channels: Number of the channels of the given data.
    :return: Reshaped data.
    """

    num_samples = data.shape[0]
    data = data.reshape(num_samples, -1)
    num_features = data.shape[-1]
    height = int(np.sqrt(num_features / num_channels))
    width = num_features // (num_channels*height)
    if not isinstance(width, int):
        raise ValueError('\nThe input images should be in square form...')

    if data_shape == 'vector':
        pass

    elif data_shape == 'matrix':
        if num_channels == 1:
            data = data.reshape(num_samples, height, width)
        elif num_channels == 3:
            data = data.reshape(num_samples, height, width, num_channels)
            # Transform RGB images into gray-scale images
            data = 0.2989 * data[:, :, :, 0] + 0.5870 * data[:, :, :, 1] + 0.1140 * data[:, :, :, 2]
            data = data.reshape(num_samples, height, width)
        else:
            raise ValueError('The input data should be either gray-scale images or color images...')

    elif data_shape == 'tensor':
        data = data.reshape(num_samples, height, width, num_channels)

    else:
        raise ValueError('\nNo suitable data shape is found. Please enter a desired data shape...')

    return data


def get_data(dataset=None, normal_class=None, data_format=None, preprocess='minmax'):
    """Obtain the dataset in a desired form stored in a dictionary.

    :param dataset: The name of desired dataset: mnist, fmnist or cifar10.
    :param normal_class: The class which is considered to be known during training.
    :param data_format: The desired data shape: vector, matrix or tensor.
    :param preprocess: The name of a preprocessing method: minmax or mean.
    :return: A dictionary containing training and testing samples.
    """

    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        num_channel = 1
    elif dataset == 'fmnist':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        num_channel = 1
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        num_channel = 3
    elif dataset == 'svhn':
        x_train, y_train, x_test, y_test = get_svhn()
        num_channel = 3
    else:
        raise ValueError('\nNo datasets are found. Please select one dataset...')

    # Reshape data and its label into desired format
    y_train = np.reshape(y_train, newshape=(-1,))
    y_test = np.reshape(y_test, newshape=(-1,))

    x_train = _reshape_data(data=x_train, data_shape=data_format, num_channels=num_channel)
    x_test = _reshape_data(data=x_test, data_shape=data_format, num_channels=num_channel)

    # Data normalization
    x_train = (x_train / 255).astype('float32')
    x_test = (x_test / 255).astype('float32')

    if preprocess == 'minmax':
        x_train = (x_train - np.min(x_train))/(np.max(x_train) - np.min(x_train))
        x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))
    elif preprocess == 'mean':
        x_train = x_train - np.mean(x_train)
        x_test = x_test - np.mean(x_test)
    else:
        raise ValueError('\nPlease give a valid preprocessing method...')

    if normal_class is None:
        data = {'x_train': x_train,
                'y_train': y_train,
                'x_test': x_test,
                'y_test': y_test}
    else:
        train_set = _extract_data(data=x_train, label=y_train, target_lb=normal_class)
        test_set = _extract_data(data=x_test, label=y_test, target_lb=normal_class)

        data = {'x_train_normal': train_set[0], 'x_train_abnormal': train_set[1],
                'y_train_normal': train_set[2], 'y_train_abnormal': train_set[3],
                'x_test_normal': test_set[0], 'x_test_abnormal': test_set[1],
                'y_test_normal': test_set[2], 'y_test_abnormal': test_set[3]}
    return data


# ==================== Image Processing ====================


def assign_label(normal_class=None, original_label=None, include_zero=None):
    """Assign labels to the selected known classes.

    :param normal_class: A list of unique selected known classes labels.
    :param original_label: A list of selected known classes samples' labels-
    :param include_zero: Boolean variable. True for include zero as the starting label. Otherwise one.
    :return: Modified labels.
    """

    num_normal_cls = len(normal_class)

    temp_lb = 0
    for idx in range(num_normal_cls):
        temp_lb = temp_lb + (original_label == normal_class[idx]) * idx

    # New labels begin with 1
    if not include_zero:
        temp_lb = temp_lb + 1
    return temp_lb


def split_data(model_name=None, data=None, rho=None, split_method=None, ground_truth=None, normal_class=None):
    """Split the dataset according to a given split method.

    :param model_name: The name for the model used for splitting.
    :param data: Selected known classes training data.
    :param rho: Splitting ratio.
    :param split_method: The name of splitting method: cnn.
    :param ground_truth: Original label list for the selected training samples.
    :param normal_class: A list of selected known classes.
    :return: Indices of typical and atypical samples in the training dataset.
    """

    cnn_path = './trained_models/cnn_for_ds_%s.h5' % model_name

    print('\nSplitting data...')

    if split_method == 'cnn':
        if os.path.isfile(cnn_path):
            model = load_model(filepath=cnn_path, compile=False)
        else:
            raise ValueError('No suitable CNN for data splitting...')

        print('\nCalculating categorical probability using trained CNN...')
        probability = model.predict(data, batch_size=128)
        pred = to_categorical(np.argmax(probability, axis=-1), num_classes=probability.shape[-1])

        gt = assign_label(normal_class=normal_class, original_label=ground_truth, include_zero=True)
        gt = to_categorical(gt, num_classes=len(normal_class))
        sim_score = gt * pred * probability

        sim_score = np.max(sim_score, axis=-1)
        sim_thr = np.percentile(sim_score, rho)

        print('\nThe sim_thr is %.4f' % sim_thr)
    else:
        raise ValueError('No suitable data splitting method...')

    typical_index = np.where(sim_score > sim_thr)
    atypical_index = np.where(sim_score <= sim_thr)

    return typical_index, atypical_index



