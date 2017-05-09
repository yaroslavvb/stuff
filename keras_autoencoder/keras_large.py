#!/usr/bin/env python
# Large MNIST autoencoder
import argparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=128, help='batchsize to use')
args = parser.parse_args()

from keras import optimizers
from weightnorm import SGDWithWeightnorm
from weightnorm import AdamWithWeightnorm

import util as u
import numpy as np
import scipy

from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
import numpy
import sys

prefix = "keras_large"  # large batch
epochs = 20000
# 2 sec per epoch, 12 hours


from keras import optimizers
import load_MNIST

from keras.models import Sequential
from keras.layers import Dense
from keras import callbacks
import numpy


class TestCallback(callbacks.Callback):
  def __init__(self, data_train, data_test, fn):
    print("Creating callback")
    self.data_train = data_train
    self.data_test = data_test
    self.losses_train = []
    self.losses_test = []
    self.times = []
    self.fn = fn

  def on_epoch_end(self, epoch, logs={}):
    global start_time
    x_test, y_test = self.data_test
    x_train, y_train = self.data_train
    pixel_loss_test, acc1 = self.model.evaluate(x_test, y_test, verbose=0)
    pixel_loss_train, acc2 = self.model.evaluate(x_train, y_train, verbose=0)
    loss_test = pixel_loss_test*28*28
    loss_train = pixel_loss_train*28*28
    self.losses_train.append(loss_train)
    self.losses_test.append(loss_test)
    
    import socket
    host = socket.gethostname().split('.',1)[0]
    outfn = 'data/%s_%s'%(host, self.fn)

    if epoch == 0:
      os.system("rm -f "+outfn)
    with open(outfn, "a") as myfile:
      elapsed = time.time()-start_time
      print('\n%d sec: Loss train: %.2f'%(elapsed,loss_train))
      print('%d sec: Loss test: %.2f'%(elapsed,loss_test))
      myfile.write('%d, %f, %f, %f\n'%(epoch, elapsed, loss_train, loss_test))

start_time = 0
if __name__=='__main__':
  numpy.random.seed(0)
  
  dsize = 10000


  from keras.datasets import mnist
  (X_train, y_train), (X_test, y_test) = mnist.load_data()
  X_train = X_train.astype(np.float32)
  X_train = X_train.reshape((X_train.shape[0], -1))
  X_train = X_train[:dsize,:]
  X_test = X_test.astype(np.float32)
  X_test = X_test.reshape((X_test.shape[0], -1))
  X_train /= 255
  X_test /= 255

  optimizer = AdamWithWeightnorm()

  model = Sequential()
  model.add(Dense(1024, input_dim=28*28, activation='relu'))
  model.add(Dense(1024, activation='relu'))
  model.add(Dense(1024, activation='relu'))
  model.add(Dense(196, activation='relu'))
  model.add(Dense(1024, activation='relu'))
  model.add(Dense(1024, activation='relu'))
  model.add(Dense(1024, activation='relu'))
  model.add(Dense(28*28, activation='relu'))
  model.compile(loss='mean_squared_error', optimizer=optimizer,
                metrics=[metrics.mean_squared_error])
  # nb_epochs in older version
  cb = TestCallback((X_train, X_train), (X_test,X_test), prefix)

  start_time = time.time()
  result = model.fit(X_train, X_train, validation_data=(X_test, X_test), 
                     batch_size=args.batch_size,
                     nb_epoch=epochs,
                     callbacks=[cb])

  acc_hist = np.asarray(result.history['mean_squared_error'])*28*28  # avg pixel loss->avg image image loss
  u.dump(acc_hist, "%s_losses.csv"%(prefix,))
  u.dump(cb.losses, "%s_vlosses.csv"%(prefix,))

