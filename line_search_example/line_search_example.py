"""Example of doing backtracking line-search on MNIST autoencoder.
Needs:
tensorflow
numpy
keras

"""

import numpy as np
import math
import time

import os, sys

import tensorflow as tf
import util as u
from util import t  # transpose

from util import t  # transpose
from util import c2v
from util import v2c
from util import v2c_np
from util import v2r
from util import kr  # kronecker
from util import Kmat # commutation matrix

def W_uniform(s1, s2):
  # sample two s1,s2 matrices 
  r = np.sqrt(6) / np.sqrt(s1 + s2 + 1)
  return np.random.random(2*s2*s1)*2*r-r


if __name__=='__main__':
  np.random.seed(0)
  tf.set_random_seed(0)
  dtype = np.float32
  
  from keras.datasets import mnist
  (X_train, y_train), (X_test, y_test) = mnist.load_data()
  X_train = X_train.astype(np.float32)
  X_train = X_train.reshape((X_train.shape[0], -1))
  X_test = X_test.astype(np.float32)
  X_test = X_test.reshape((X_test.shape[0], -1))
  X_train /= 255
  X_test /= 255

  

  dsize = 100
  #patches = train_images[:,:dsize];
  patches = X_train[:dsize,:].T
  
  fs = [dsize, 28*28, 196, 28*28]

  fs=fs
  X0=patches
  lambda_=3e-3
  rho=0.1
  beta=3
  W0f=None
  
  if not W0f:
    W0f = W_uniform(fs[2],fs[3])
  rho = tf.constant(rho, dtype=dtype)

  def f(i): return fs[i+1]  # W[i] has shape f[i] x f[i-1]
  dsize = f(-1)
  n = len(fs) - 2

  init_dict = {}
  def init_var(val, name, trainable=False):
    val = np.array(val)
    holder = tf.placeholder(dtype, shape=val.shape, name=name+"_holder")
    var = tf.Variable(holder, name=name+"_var", trainable=trainable)
    init_dict[holder] = val
    return var

  lr = init_var(0.01, "lr")
  Wf = init_var(W0f, "Wf", True)
  Wf_copy = init_var(W0f, "Wf_copy")
  W = u.unflatten(Wf, fs[1:])
  X = init_var(X0, "X")
  W.insert(0, X)

  def sigmoid(x):
    return tf.sigmoid(x)
  def d_sigmoid(y):
    return y*(1-y)
  def kl(x, y):
    return x * tf.log(x / y) + (1 - x) * tf.log((1 - x) / (1 - y))
  def d_kl(x, y):
    return (1-x)/(1-y) - x/y
  
  # A[i] = activations needed to compute gradient of W[i]
  # A[n+1] = network output
  A = [None]*(n+2)
  A[0] = u.Identity(dsize, dtype=dtype)
  A[1] = W[0]
  for i in range(1, n+1):
    A[i+1] = sigmoid(W[i] @ A[i])
    

  # reconstruction error and sparsity error
  err = (A[3] - A[1])
  rho_hat = tf.reduce_sum(A[2], axis=1, keep_dims=True)/dsize

  # B[i] = backprops needed to compute gradient of W[i]
  B = [None]*(n+1)
  B[n] = err*d_sigmoid(A[n+1])
  for i in range(n-1, -1, -1):
    backprop = t(W[i+1]) @ B[i+1]
    if i == 1:
      backprop += beta*d_kl(rho, rho_hat)
    B[i] = backprop*d_sigmoid(A[i+1])

  # dW[i] = gradient of W[i]
  dW = [None]*(n+1)
  for i in range(n+1):
    dW[i] = (B[i] @ t(A[i]))/dsize

  # Cost function
  reconstruction = u.L2(err) / (2 * dsize)
  sparsity = beta * tf.reduce_sum(kl(rho, rho_hat))
  L2 = (lambda_ / 2) * (u.L2(W[1]) + u.L2(W[1]))
  cost = reconstruction + sparsity + L2

  grad = u.flatten(dW[1:])
  copy_op = Wf_copy.assign(Wf-lr*grad)
  with tf.control_dependencies([copy_op]):
    train_op = tf.group(Wf.assign(Wf_copy)) # to make it an op

  sess = tf.InteractiveSession()

  #  step_len = init_var(tf.constant(0.1), "step_len", False)
  #  step_len_assign = step_len.assign(step_len0)
  step_len0 = tf.placeholder(dtype, shape=())
  
  Wf2 = init_var(W0f, "Wf2")
  Wf_save_op = Wf2.assign(Wf)
  Wf_restore_op = Wf.assign(Wf2)
  grad2 = init_var(W0f, "grad2")
  grad_save_op = grad2.assign(grad)
  grad2_norm_op = tf.reduce_sum(tf.square(grad2))
  Wf_step_op = Wf.assign(Wf2 - step_len0*grad2)
  lr_p = tf.placeholder(lr.dtype, lr.shape)
  lr_set = lr.assign(lr_p)

  def save_wf(): sess.run(Wf_save_op)
  def restore_wf(): sess.run(Wf_restore_op)
  def save_grad(): sess.run(grad_save_op)
  def step_wf(step):
    #    sess.run(step_len_assign, feed_dict={step_len0: step})
    sess.run(Wf_step_op, feed_dict={step_len0: step}) 
  
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)

  
  print("Running training.")
  do_images = True
  u.reset_time()
  old_cost = sess.run(cost)
  old_i = 0
  frame_count = 0

  step_lengths = []
  costs = []
  ratios = []
  do_bt = False  # do backtracking line-search
  alpha=0.3
  beta=0.8
  growth_rate = 1.05
  for i in range(10000):
    # save Wf and grad into Wf2 and grad2
    save_wf()
    save_grad()
    cost0 = cost.eval()
    train_op.run()
    lr0 = lr.eval()
    cost1 = cost.eval()
    #    cost1, _ = sess.run([cost, train_op])
    target_delta = -alpha*lr0*grad2_norm_op.eval()
    expected_delta = -lr0*grad2_norm_op.eval()
    actual_delta = cost1 - cost0
    actual_slope = actual_delta/lr0
    expected_slope = -grad2_norm_op.eval()

    # ratio of best possible slope to actual slope
    # don't divide by actual slope because that can be 0
    slope_ratio = abs(actual_slope)/abs(expected_slope)
    costs.append(cost0)
    step_lengths.append(lr0)
    ratios.append(slope_ratio)

    if i%10 == 0:
      print("Learning rate: %f"% (lr0,))
      print("Cost %.2f, expected decrease %.2f, actual decrease, %.2f ratio %.2f"%(cost0, expected_delta, actual_delta, slope_ratio))

    # don't shrink learning rate once results are very close to minimum
    if slope_ratio < alpha and abs(target_delta)>1e-6:
      print("%.2f %.2f %.2f"%(cost0, cost1, slope_ratio))
      print("Slope optimality %.2f, shrinking learning rate to %.2f"%(slope_ratio, lr0*beta,))
      sess.run(lr_set, feed_dict={lr_p: lr0*beta})
    else:
      # see if our learning rate got too conservative, and increase it
      if i>0 and i%10 == 0 and slope_ratio>0.99:
        print("%.2f %.2f %.2f"%(cost0, cost1, slope_ratio))
        print("Growing learning rate to %.2f"%(lr0*growth_rate))
        sess.run(lr_set, feed_dict={lr_p: lr0*growth_rate})

    u.record_time()

  u.dump(step_lengths, "step_lengths_ada.csv")
#  u.dump(costs, "costs_ada.csv")
#  u.dump(ratios, "ratios_ada.csv")
