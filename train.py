import glob
import argparse
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from net_model import *

import imageio
import os
from os.path import isfile, join, isdir
import shutil
import datetime
import h5py

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--model_path", required=True)
    return parser.parse_args()


def main():
    EPOCHS = 300
    BATCH_SIZE = 32
    PERC_TRAIN = 0.96
    input_bgr = True
    max_shape = [256, 512]
    n_gpus = 1

    folder_name = '{date:%Y%m%d_%H%M}'.format( date=datetime.datetime.now())

    args = parse_args()

    h5_file = h5py.File(args.dataset_path, 'r', swmr=True)
    dataset_size = len(h5_file.keys())

    print(tf.__version__)
    train_size = round(dataset_size*PERC_TRAIN)

    model = get_model(input_bgr, n_gpus)
    os.makedirs(join(args.model_path,folder_name))
    os.makedirs(join(args.model_path,folder_name,"eval"))
    os.makedirs(join(args.model_path,folder_name,"models"))
    keras.utils.plot_model(model, join(args.model_path,folder_name,'model.png'), show_shapes=True)

    model.summary()
    for layer in model.layers:
        print(layer.name, ':', layer.trainable)

    log_dir = join(args.model_path,folder_name,"logs")
    shutil.rmtree(log_dir, ignore_errors=True)

    tensorboard_cbk = keras.callbacks.TensorBoard(log_dir=log_dir)
    model_saver =  keras.callbacks.ModelCheckpoint(filepath=join(args.model_path,folder_name,"models","best_model.h5"),save_best_only=True,monitor='val_loss', mode='min', verbose=1)
    earlystopper = keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',patience=50, verbose=1)

    history = model.fit_generator(get_generator(h5_file, BATCH_SIZE, range(train_size), max_shape, input_bgr),
    validation_data=get_generator(h5_file, BATCH_SIZE, range(train_size,dataset_size), max_shape, input_bgr), steps_per_epoch=train_size/BATCH_SIZE, validation_steps=1, epochs=EPOCHS, callbacks=[tensorboard_cbk,model_saver,earlystopper])

    model.save(join(args.model_path,folder_name,'models','final_model.h5'))

    print('\nhistory dict:', history.history)

    subset = [0,3,5,900,902,990,995]
    valX, valY = get_data(h5_file, subset, max_shape, input_bgr)
    predictions = model.predict(valX)
    Xim = get_im(dset_im, subset)
    
    for ti in range(predictions.shape[0]):
        fname = "%08d_" % (ti)
        im = Xim[ti,:,:,:3]
        H = predictions[ti,:,:,0]
        npmax = np.max(H)
        if npmax == 0:
            npmax = 1
        imageio.imwrite(join(args.model_path,folder_name,"eval",fname+"im.png"), (255*im).astype(np.uint8))
        imageio.imwrite(join(args.model_path,folder_name,"eval",fname+"psf.png"), (255*(H/npmax)).astype(np.uint8))
        # imageio.imwrite(join(args.model_path,"eval",fname+"bgr.png"), (255*B).astype(np.uint8))
        # imageio.imwrite(join(args.model_path,"eval",fname+"F.png"), (255*Fsave).astype(np.uint8))
        # imageio.imwrite(join(args.model_path,"eval",fname+"M.png"), (255*M).astype(np.uint8))

if __name__ == "__main__":
    main()