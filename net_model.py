from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Model    
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K

import numpy as np
import cv2
import skimage.transform
import random
from dataset.helpers import *
import scipy

def conv_layer(x, n_filters, do_batchnorm = False, activation = None):
    for kk in range(3):
        x = Conv2D(n_filters, (3,3), activation = activation, padding='same')(x) 
        x = LeakyReLU(alpha=0.1)(x)
        if do_batchnorm: x = BatchNormalization()(x)

    return x

def get_model(input_bgr, n_gpus=1):
    if input_bgr:
        input_layer = Input(shape=(None, None, 6))
    else:
        input_layer = Input(shape=(None, None, 3))

    ## encoder
    x = conv_layer(input_layer, 16)
    x_skip1 = x
    x = MaxPooling2D()(x)
    x = conv_layer(x, 64)
    x_skip2 = x
    x = MaxPooling2D()(x)
    x = conv_layer(x, 128)
    x_skip3 = x
    x = MaxPooling2D()(x)
    x = conv_layer(x, 256)
    x_skip4 = x
    x = MaxPooling2D()(x)

    ## latent space
    x = conv_layer(x, 256)

    ## decoder
    x = Conv2DTranspose(128, kernel_size=(2,2), strides=(2,2))(x)
    x = Concatenate()([x, x_skip4])
    x = conv_layer(x, 128)

    x = Conv2DTranspose(64, kernel_size=(2,2), strides=(2,2))(x)
    x = Concatenate()([x, x_skip3])
    x = conv_layer(x, 64)

    x = Conv2DTranspose(32, kernel_size=(2,2), strides=(2,2))(x)
    x = Concatenate()([x, x_skip2])
    x = conv_layer(x, 32)

    x = Conv2DTranspose(16, kernel_size=(2,2), strides=(2,2))(x)
    x = Concatenate()([x, x_skip1])
    x = conv_layer(x, 16)

    x = Conv2D(4,(3,3), activation = None, padding='same')(x) 
    x = Conv2D(4,(3,3), activation = None, padding='same')(x) 
    x = Conv2D(1,(3,3), activation = None, padding='same')(x) 
    
    output_layer = x
    model = Model(inputs = input_layer, outputs = output_layer)
    if n_gpus > 1:
        model = tf.keras.utils.multi_gpu_model(model, gpus=n_gpus)

    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    model.compile(optimizer = optimizer, loss = custom_loss(input_layer), 
                metrics=["mae"], experimental_run_tf_function=False)

    return model


def custom_loss(input_layer):
    def fmo_loss_function(Yact, Ypred):
        Ha = Yact[:,:,:,0]
        H = Ypred[:,:,:,0]
        ### DT loss inverse
        total_loss = K.mean(K.abs(H[Ha > 0] - Ha[Ha > 0])) + K.mean(K.abs(H[Ha==0] - 0))
        return total_loss

    return fmo_loss_function

def get_generator(h5_file, batch_size, all_subset, max_shape, input_bgr):
    while True:
        subset = random.sample(all_subset, batch_size)
        X, Y = get_data(h5_file, subset, max_shape, input_bgr)
        yield X, Y

def process_image(img, do_resize = True, shape = None):
    if shape is None:
        shape = img.shape[:2]
    shape = [16 * (sh // 16) for sh in shape]
    if do_resize:
        img = skimage.transform.resize(img, shape, order=3)
    else:
        img = img[:shape[0],:shape[1]]
    img = img - np.mean(img)
    img = img / np.sqrt(np.var(img))
    return img

def get_data(h5_file, subset, max_shape, input_bgr):
    if input_bgr:
        X = np.zeros([len(subset),max_shape[0],max_shape[1],6])
    else:
        X = np.zeros([len(subset),max_shape[0],max_shape[1],3])

    Y = np.zeros([len(subset),max_shape[0],max_shape[1],1])

    ki = 0
    for k in subset:
        fname = "%08d_" % (k)
        if 'im' in h5_file[fname].keys():
            I = skimage.transform.resize(h5_file[fname]['im'], max_shape, order=3)
            I = process_image(I,False)

            H = skimage.transform.resize(h5_file[fname]['psf'], max_shape, order=1)

            M = skimage.transform.resize(h5_file[fname]['M'], max_shape, order=1)

            rad = np.sqrt(np.sum(M))
            DT = scipy.ndimage.morphology.distance_transform_edt(H == 0)
            DT = DT / (2*rad)
            DT[DT > 1] = 1

            X[ki,:,:,:3] = I

            if input_bgr:
                B = skimage.transform.resize(h5_file[fname]['bgr'], max_shape, order=3)
                B = process_image(B,False)
                X[ki,:,:,3:] = B

            Y[ki,:,:,0] = 1 - DT
        else:
            X[ki] = h5_file[fname]['X']
            Y[ki] = h5_file[fname]['Y']
        ki = ki + 1
    return X,Y

def get_im(path):
    I = cv2.imread(path)
    return I