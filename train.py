"""
wontak ryu
ryu071511@gmail.com
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.python.keras.datasets import mnist, cifar10
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras import backend as K
import numpy as np
from module import mixup_data, get_model, compute_loss, compute_acc, base_compute_loss
from hyper_params import get_hyper_params
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
    mixup_model = get_model(input_shape, num_classes, dropout)
    base_model = get_model(input_shape, num_classes, dropout)

    if os.path.exists(mixup_path) and os.path.exists(base_path):
        print("loading trained model...")
        mixup_model.load_weights(mixup_path)
        base_model.load_weights(base_path)

    if dataset == 'mnist':
        (x, y), _ = mnist.load_data()
    elif dataset == 'cifar10':
        (x, y), _ = cifar10.load_data()

    x1, y_a, y_b, lam = mixup_data(x, y, alpha)
    shape = [-1] + input_shape
    x1 = np.reshape(x1, shape)
    Y = np.concatenate([y_a, y_b, lam], axis=-1)
    callback = [ModelCheckpoint(mixup_path,
                               monitor='val_loss',
                               save_best_only=True,
                               save_weights_only=True)]

    mixup_model.compile(loss=compute_loss,
                        optimizer=Adam(hyper_params['learning_rate']),
                        metrics=[compute_acc])

    print("Training model on mixup dataset...")
    mixup_model.fit(x=x1,
                    y=Y,
                    validation_split=0.2,
                    batch_size=hyper_params['batch_size'],
                    callbacks=callback,
                    epochs=hyper_params['epoch'])

    # # base model
    # callback = [ModelCheckpoint(base_path,
    #                            monitor='val_loss',
    #                            save_best_only=True,
    #                            save_weights_only=True,)]
    #
    # shape = [-1] + input_shape
    # x = np.reshape(x, shape)
    # y = tf.keras.utils.to_categorical(y)
    # base_model.compile(loss='categorical_crossentropy',
    #                    optimizer=Adam(hyper_params['learning_rate']),
    #                    metrics=['accuracy'])
    # print("Training model on original dataset...")
    # base_model.fit(x=x,
    #                y=y,
    #                validation_split=0.2,
    #                batch_size=hyper_params['batch_size'],
    #                callbacks=callback,
    #                epochs=hyper_params['epoch'])
    #
