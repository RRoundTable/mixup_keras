"""
wontak ryu
ryu071511@gmail.com
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.python.keras.datasets import mnist, cifar10, cifar100
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
import numpy as np
from module import get_model, get_deeper_model, compute_loss, compute_acc
from hyper_params import get_hyper_params
import pandas as pd
import os

# hyper parameters
hyper_params = get_hyper_params()
K.set_image_data_format('channels_last')
alpha = hyper_params['alpha']
input_shape = hyper_params['input_shape']
num_classes = hyper_params['num_classes']
dataset = hyper_params['dataset']
dropout = hyper_params['dropout']
mixup_path = "mixup_alpha{}_{}.hdf5".format(alpha, dataset)
base_path = "base_alpha{}_{}.h5".format(alpha, dataset)

if __name__ == "__main__":

    mixup_model = get_deeper_model(input_shape, num_classes, dropout)
    base_model = get_deeper_model(input_shape, num_classes, dropout)
    mixup_model.load_weights(mixup_path)
    base_model.load_weights(base_path)

    mixup_model.compile(loss=compute_loss,
                        optimizer=Adam(hyper_params['learning_rate']),
                        metrics=[compute_acc])
    base_model.compile(loss='categorical_crossentropy',
                       optimizer=Adam(hyper_params['learning_rate']),
                       metrics=['accuracy'])
    if dataset == 'mnist':
        _, (x_test, y_test) = mnist.load_data()
    elif dataset == 'cifar10':
        _, (x_test, y_test) = cifar10.load_data()
    elif dataset == 'cifar100':
        _, (x_test, y_test) = cifar100.load_data()

    shape = [-1] + input_shape
    x_test = np.reshape(x_test, shape)
    lam = np.array([1] * len(x_test))
    y_test = np.reshape(y_test, (-1, 1))
    lam = np.reshape(lam, (-1, 1))
    Y = np.concatenate([y_test, y_test, lam], axis=-1)

    y_test = tf.keras.utils.to_categorical(y_test)
    print("Evaluate model trained on mixup dataset...")
    mixup_result = mixup_model.evaluate(x_test, Y)
    print("Evaluate model trained on original dataset...")
    base_result = base_model.evaluate(x_test, y_test)


    save = hyper_params
    save['base_loss'] = base_result[0]
    save['base_acc'] = base_result[1]
    save['mixup_loss'] = mixup_result[0]
    save['mixup_acc'] = mixup_result[1]
    save = sorted(save.items(), key=(lambda x: x[0]))
    dataframe = pd.DataFrame(save)
    dataframe['base_loss'] = base_result[0]
    dataframe['base_acc'] = base_result[1]
    dataframe['mixup_loss'] = mixup_result[0]
    dataframe['mixup_acc'] = mixup_result[1]
    dataframe.to_csv("./experiments/alpha_{}dropout_{}dataset_{}.csv".format(alpha, dropout, dataset ),
                     encoding='utf-8')


