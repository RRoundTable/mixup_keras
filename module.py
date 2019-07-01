import tensorflow as tf

from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras import backend as K
from hyper_params import get_hyper_params
import numpy as np

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

def mixup_data(x1, y1, alpha):
    """Mix data
    x1: input numpy array.
    y1: target numpy array.
    alpha: float.
    Paper url: https://arxiv.org/pdf/1710.09412.pdf
    See Figure1.
    Return
        mixed_x, y_a, y_b, lam
    """
    n = len(x1)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha, n)
    else:
        lam = np.array([1.0] * n)
    indexs = np.random.randint(0, n, n)
    for i in range(n):
        x1[i] = x1[i] * lam[i] + (1 - lam[i]) * x1[indexs[i]]
    y_a = np.reshape(y1, (-1, 1))
    y_b = np.reshape(y1[indexs], (-1, 1))
    lam = np.reshape(lam, (-1, 1))
    if len(input_shape) == 2:
        shape = [-1] + input_shape + [1]
        x1 = np.reshape(x1, shape)
    return x1, y_a, y_b, lam

def mixup_data_one_sample(x1, y1, x2, y2, lam):
    """Mix data on one sample.
    x1: input numpy array.
    y1: target numpy array.
    lam: lambda. [0, 1]
    Paper url: https://arxiv.org/pdf/1710.09412.pdf
    See Figure1.
    Return
        mixed_x, y_a, y_b, lam
    """
    x = x1 * lam + x2 * (1.0 - lam)
    return x, y1, y2, lam

def get_model(input_shape, num_classes, dropout):
    """Return keras model"""
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(num_classes, activation='softmax', name="main"))
    return model

def get_deeper_model(input_shape, num_classes, dropout, weight_decay=1e-4):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='relu',
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Conv2D(64, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Conv2D(128, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same',
                     kernel_regularizer=regularizers.l2(weight_decay),
                     activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    return model

# loss function
def compute_loss(y_true, y_pred):
    y_a, y_b, lam = tf.split(y_true, 3, 1)
    y_a = tf.cast(y_a, dtype=tf.int32)
    y_b = tf.cast(y_b, dtype=tf.int32)
    y_a = tf.reshape(y_a, (-1,))
    y_b = tf.reshape(y_b, (-1,))
    lam = tf.reshape(lam, (-1,))
    y_a = K.one_hot(y_a, num_classes)
    y_b = K.one_hot(y_b, num_classes)
    return tf.add(tf.math.multiply(lam, categorical_crossentropy(y_a, y_pred)),
                  tf.math.multiply((1 - lam), categorical_crossentropy(y_b, y_pred)))

def base_compute_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.int32)
    y_true = K.one_hot(y_true, num_classes)
    return categorical_crossentropy(y_true, y_pred)

def compute_acc(y_true, y_pred):
    y_a, y_b, lam = tf.split(y_true, 3, 1)
    y_a = tf.cast(y_a, tf.int64)
    y_b = tf.cast(y_b, tf.int64)
    y_a = tf.reshape(y_a, (-1,))
    y_b = tf.reshape(y_b, (-1,))
    y_pred = tf.argmax(y_pred, axis=-1)
    return lam * accuracy(y_a, y_pred) + (1 - lam) * accuracy(y_b, y_pred)