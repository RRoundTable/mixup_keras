"""
wontak ryu
ryu071511@gmail.com
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.python.keras.applications import *
from tensorflow.python.keras.datasets import mnist, cifar100
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras import backend as K
import numpy as np

# tf.enable_eager_execution()
# print("eager mode: {}".format(tf.executing_eagerly()))

K.set_image_data_format('channels_last')
alpha = 1.0
input_shape = [28, 28, 1]
num_classes = 10

# optimizer

optimizer = tf.optimizers.Adam(0.001)

def mixup_data(x1, y1, alpha):
    """Mix data
    x1: input numpy array.
    y1: target numpy array.
    alpha: float.
    Paper url: https://arxiv.org/pdf/1710.09412.pdf
    See Figure1.
    Return
        mixed_x, mixed_y
    """
    n = len(x1)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    indexs = np.random.randint(0, n, n)
    x1 = np.array(x1 * lam + (1 - lam) * x1[indexs])
    y_a = y1
    y_b = y1[indexs]
    lam = [lam] * n
    return x1, y_a, y_b, lam

def mixup_data1(x1, y1, alpha):
    """Mix data
    x1: input numpy array.
    y1: target numpy array.
    alpha: float.
    Paper url: https://arxiv.org/pdf/1710.09412.pdf
    See Figure1.
    Return
        mixed_x, mixed_y
    """
    n = len(x1)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha, n)
    else:
        lam = np.array([1.0] * n)
    indexs = np.random.randint(0, n, n)
    for i in range(n):
        if i % 100 == 0:
            print("mixup")
        x1[i] = x1[i] * lam[i] + (1 - lam[i]) * x1[indexs[i]]
    y_a = y1
    y_b = y1[indexs]
    return x1, y_a, y_b, lam

def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

loss_func = categorical_crossentropy
# loss function
def compute_loss(y_a, y_b, y_pred, lam):
    return lam * loss_func(y_a, y_pred) + (1 - lam) * loss_func(y_b, y_pred)

# acc function
acc_func = accuracy
def compute_acc(y_a, y_b, y_pred, lam):
    y_a = tf.argmax(y_a, axis=-1)
    y_b = tf.argmax(y_b, axis=-1)
    return lam * acc_func(y_a, y_pred) + (1 - lam) * acc_func(y_b, y_pred)

def _prepare_mnist_features_and_labels_mixup(x, y_a, y_b, lam):
    x = tf.cast(x, tf.float32) / 255.0
    y_a = tf.cast(y_a, tf.int64)
    y_b = tf.cast(y_b, tf.int64)
    lam = tf.cast(lam, tf.float32)
    return x, y_a, y_b, lam

def _prepare_mnist_features_and_labels(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y

def mnist_mixup_dataset():
    (x, y), _ = tf.keras.datasets.mnist.load_data()
    x = np.reshape(x, [60000, 28, 28, 1])
    x, y_a, y_b, lam = mixup_data(x, y, alpha)
    y_a = tf.reshape(y_a, [-1, 1])
    y_b = tf.reshape(y_b, [-1, 1])
    y_a = K.one_hot(y_a, num_classes)
    y_b = K.one_hot(y_b, num_classes)
    y_a = tf.reshape(y_a, [-1, 10])
    y_b = tf.reshape(y_b, [-1, 10])
    ds = tf.data.Dataset.from_tensor_slices((x, y_a, y_b, lam))
    ds = ds.map(_prepare_mnist_features_and_labels_mixup)
    ds = ds.take(20000).shuffle(20000).batch(100)
    return ds

def mnist_dataset():
    (x, y), _ = tf.keras.datasets.mnist.load_data()
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(_prepare_mnist_features_and_labels)
    ds = ds.take(20000).shuffle(20000).batch(100)
    return ds

@tf.function
def train_one_step(model, optimizer, x, y_a, y_b, lam):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(y_a, y_b, logits, lam)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    y_pred = tf.argmax(logits, axis=-1)
    acc = compute_acc(y_a, y_b, y_pred, lam)
    return loss, acc

@tf.function
def train(epoch, model, optimizer, train_ds):
    loss = 0.0
    acc = 0.0
    for i in range(epoch):
        for x, y_a, y_b, lam in train_ds:
            loss, acc = train_one_step(model, optimizer, x, y_a, y_b, lam)
        if i % 10 == 0:
            tf.print("Step: {} Loss: {} ACC: {}".format(i, loss, acc))
    return i, loss, acc


if __name__ == "__main__":

    # # baseline
    # base_model = get_model()
    # base_ds = mnist_dataset()
    # train(100, base_model, optimizer, base_ds)

    # mixup
    mixup_model = get_model()
    mixup_ds = mnist_mixup_dataset()
    train(100, mixup_model, optimizer, mixup_ds)
    print("end")


