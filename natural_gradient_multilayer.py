#!/bin/env python -u
#
# Apply natural gradient (using empirical Fisher) to a multilayer problem
# Accompanying notebooks:
#
# natural_gradient_multilayer.nb
# https://www.wolframcloud.com/objects/a273119a-6eb0-4521-b79d-30795f155dc4
# 
# relus.nb
# https://www.wolframcloud.com/objects/b05dd44c-c9da-4187-831b-32eebb7a5d02


import numpy as np
import sys
import tensorflow as tf
import traceback


from util import *

dtype = np.float64

# convention, X0 is numpy, X is Tensor
def gd_test():
  """Test gradient descent, using tf.gradients for backprop."""
  
  tf.reset_default_graph()

  # load data into TF
  XY0 = np.genfromtxt('data/natural_gradient_multilayer_XY0.csv',
                      delimiter= ",")
  
  fs = np.genfromtxt('data/natural_gradient_multilayer_fs.csv',
                     delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers
  
  X0 = XY0[:-1,:]  # 2 x d
  Y0 = XY0[-1:,:]  # 1 x d
  dsize = X0.shape[1]
  Y_ = tf.placeholder(dtype, shape=Y0.shape, name="Y_holder")
  Y = tf.Variable(Y_, trainable=False)
  init_dict={Y_: Y0}

  W0f = v2c_np(np.genfromtxt('data/natural_gradient_multilayer_W0f.csv',
                             delimiter= ","))
  W0s = unflatten_np(W0f, fs[1:])  # Wf doesn't have first layer (data matrix)
  W0s.insert(0, X0)

  
  # initialize data + layers
  W = []   # list of "W" matrices. W[0] is input matrix (X), W[n] is last matrix
  Wi_holders = []
  A = [Identity(dsize)]   # activation matrices
  for i in range(n+1):
    # fs is off by 2 from common notation, ie W[0] has shape f[0],f[-1]
    Wi_name = "W"+str(i)
    Wi_shape = (fs[i+1], fs[i])
    Wi_holder = tf.placeholder(dtype, shape=Wi_shape, name=Wi_name+"_h")
    Wi_holders.append(Wi_holder)  # TODO: delete
    Wi = tf.Variable(Wi_holder, name=Wi_name, trainable=(i>0))
    Ai_name = "A"+str(i+1)
    Ai = tf.matmul(Wi, A[-1], name=Ai_name)
    A.append(Ai)
    W.append(Wi)
    
    init_dict[Wi_holder] = W0s[i]

  assert len(A) == n+2
  
  assert W[0].shape == (2, 10)
  assert W[1].shape == (2, 2)
  assert W[2].shape == (2, 2)
  assert W[3].shape == (1, 2)

  assert X0.shape == (2, 10)
  assert W0s[1].shape == (2, 2)
  assert W0s[2].shape == (2, 2)
  assert W0s[3].shape == (1, 2)
  
  assert A[0].shape == (10, 10)
  assert A[1].shape == (2, 10)
  assert A[2].shape == (2, 10)
  assert A[3].shape == (2, 10)
  assert A[4].shape == (1, 10)

  
  # input dimensions match
  assert W[0].get_shape() == X0.shape
  # output dimensions match
  assert W[-1].get_shape()[0], W[0].get_shape()[1] == Y0.shape
  assert A[n+1].get_shape() == Y0.shape

  err = Y - A[n+1]
  #  loss = (1./(2*dsize))*(err @ tf.transpose(err))
  loss = tf.reduce_sum(tf.square(err))/(2*dsize)
  lr = 0.5
  
  opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
  grads_and_vars = opt.compute_gradients(loss, tf.trainable_variables())
  train_op = opt.apply_gradients(grads_and_vars)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  expected_losses = np.loadtxt("data/natural_gradient_multilayer_losses_gd.csv")
  
  observed_losses = []
  for i in range(20):
    observed_losses.append(sess.run([loss])[0])
    sess.run(train_op)

  check_equal(observed_losses, expected_losses)

# convention, X0 is numpy, X is Tensor
def gd_manual_test():
  """Train network, without using tf.gradients"""
  
  tf.reset_default_graph()

  # load data into TF
  XY0 = np.genfromtxt('data/natural_gradient_multilayer_XY0.csv',
                      delimiter= ",")
  
  fs = np.genfromtxt('data/natural_gradient_multilayer_fs.csv',
                     delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers
  
  X0 = XY0[:-1,:]  # 2 x d
  Y0 = XY0[-1:,:]  # 1 x d
  dsize = X0.shape[1]
  Y_ = tf.placeholder(dtype, shape=Y0.shape, name="Y_holder")
  Y = tf.Variable(Y_, trainable=False)
  init_dict={Y_: Y0}

  W0f = v2c_np(np.genfromtxt('data/natural_gradient_multilayer_W0f.csv',
                             delimiter= ","))
  W0s = unflatten_np(W0f, fs[1:])  # Wf doesn't have first layer (data matrix)
  W0s.insert(0, X0)

  
  # initialize data + layers
  # W[0] is input matrix (X), W[n] is last matrix
  # A[1] has activations for W[1], equal to W[0]=X
  # A[n+1] has predictions
  W = [0]*(n+1)   # list of "W" matrices. 
  A = [0]*(n+2)
  A[0] = Identity(dsize)
  for i in range(n+1):
    # fs is off by 2 from common notation, ie W[0] has shape f[0],f[-1]
    Wi_name = "W"+str(i)
    Wi_shape = (fs[i+1], fs[i])
    Wi_holder = tf.placeholder(dtype, shape=Wi_shape, name=Wi_name+"_h")
    W[i] = tf.Variable(Wi_holder, name=Wi_name, trainable=(i>0))
    A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))
    init_dict[Wi_holder] = W0s[i]

  assert len(A) == n+2
  
  assert W[0].shape == (2, 10)
  assert W[1].shape == (2, 2)
  assert W[2].shape == (2, 2)
  assert W[3].shape == (1, 2)

  assert X0.shape == (2, 10)
  assert W0s[1].shape == (2, 2)
  assert W0s[2].shape == (2, 2)
  assert W0s[3].shape == (1, 2)
  
  assert A[0].shape == (10, 10)
  assert A[1].shape == (2, 10)
  assert A[2].shape == (2, 10)
  assert A[3].shape == (2, 10)
  assert A[4].shape == (1, 10)

  
  # input dimensions match
  assert W[0].get_shape() == X0.shape
  # output dimensions match
  assert W[-1].get_shape()[0], W[0].get_shape()[1] == Y0.shape
  assert A[n+1].get_shape() == Y0.shape

  err = Y - A[n+1]
  #  loss = (1./(2*dsize))*(err @ tf.transpose(err))
  loss = tf.reduce_sum(tf.square(err))/(2*dsize)
  lr = tf.Variable(0.5, dtype=dtype)
  
  # create backprop matrices
  # B[i] has backprop for matrix i
  B = [0]*(n+1)
  B[n] = -err/dsize
  for i in range(n-1, -1, -1):
    B[i] = tf.matmul(tf.transpose(W[i+1]), B[i+1], name="B"+str(i))

  # Create gradient update. Make copy of variables and split update into
  # two run calls. Using single set of variables will gives updates that 
  # occasionally produce wrong results/NaN's because of data race
  
  dW = [0]*(n+1)
  updates1 = [0]*(n+1)  # compute updated value into Wcopy
  updates2 = [0]*(n+1)  # copy value back into W
  Wcopy = [0]*(n+1)
  for i in range(n+1):
    Wi_name = "Wcopy"+str(i)
    Wi_shape = (fs[i+1], fs[i])
    Wi_init = tf.zeros(dtype=dtype, shape=Wi_shape, name=Wi_name+"_init")
    Wcopy[i] = tf.Variable(Wi_init, name=Wi_name, trainable=False)
    
    dW[i] = tf.matmul(B[i], tf.transpose(A[i]), name="dW"+str(i))
    updates1[i] = Wcopy[i].assign(W[i]-lr*dW[i])
    updates2[i] = W[i].assign(Wcopy[i])

  del updates1[0]  # don't update input matrices
  del updates2[0]
  
  train_op1 = tf.group(*updates1)
  train_op2 = tf.group(*updates2)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  expected_losses = np.loadtxt("data/natural_gradient_multilayer_losses_gd.csv")
  
  observed_losses = []
  for i in range(20):
    observed_losses.append(sess.run([loss])[0])
    sess.run(train_op1)
    sess.run(train_op2)

  check_equal(observed_losses, expected_losses)

# convention, X0 is numpy, X is Tensor
def gd_manual_vectorized_test():
  """Train network, with manual backprop, in vectorized form"""
  
  tf.reset_default_graph()

  # load data into TF
  XY0 = np.genfromtxt('data/natural_gradient_multilayer_XY0.csv',
                      delimiter= ",")
  
  fs = np.genfromtxt('data/natural_gradient_multilayer_fs.csv',
                     delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers
  
  X0 = XY0[:-1,:]  # 2 x d
  Y0 = XY0[-1:,:]  # 1 x d
  dsize = X0.shape[1]
  Y_ = tf.placeholder(dtype, shape=Y0.shape, name="Y_holder")
  Y = tf.Variable(Y_, trainable=False)
  init_dict={Y_: Y0}

  W0f = v2c_np(np.genfromtxt('data/natural_gradient_multilayer_W0f.csv',
                             delimiter= ","))
  W0s = unflatten_np(W0f, fs[1:])  # Wf doesn't have first layer (data matrix)
  W0s.insert(0, X0)

  
  # initialize data + layers
  # W[0] is input matrix (X), W[n] is last matrix
  # A[1] has activations for W[1], equal to W[0]=X
  # A[n+1] has predictions
  W = [0]*(n+1)   # list of "W" matrices. 
  A = [0]*(n+2)
  A[0] = Identity(dsize)

  dims = [(fs[i+1],fs[i]) for i in range(len(fs)-1)]
  sizes = [s[0]*s[1] for s in dims]
  Wf_size = np.sum(sizes[1:])
  Wf_holder = tf.placeholder(dtype, shape=W0f.shape)
  Wf = tf.Variable(Wf_holder, name="Wf")
  assert Wf.shape == W0f.shape
  init_dict[Wf_holder] = W0f
  
  W = unflatten(Wf, fs[1:])
  W.insert(0, tf.constant(X0))
  assert W[0].shape == [2, 10]
  assert W[1].shape == [2, 2]
  assert W[2].shape == [2, 2]
  assert W[3].shape == [1, 2]
                   
  for i in range(n+1):
    # fs is off by 2 from common notation, ie W[0] has shape f[0],f[-1]
    A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))
    
  assert len(A) == n+2
  assert A[0].shape == (10, 10)
  assert A[1].shape == (2, 10)
  assert A[2].shape == (2, 10)
  assert A[3].shape == (2, 10)
  assert A[4].shape == (1, 10)

  
  # input dimensions match
  assert W[0].get_shape() == X0.shape
  # output dimensions match
  assert W[-1].get_shape()[0], W[0].get_shape()[1] == Y0.shape
  assert A[n+1].get_shape() == Y0.shape

  err = Y - A[n+1]
  #  loss = (1./(2*dsize))*(err @ tf.transpose(err))
  loss = tf.reduce_sum(tf.square(err))/(2*dsize)
  lr = tf.Variable(0.5, dtype=dtype)
  
  # create backprop matrices
  # B[i] has backprop for matrix i
  B = [0]*(n+1)
  B[n] = -err/dsize
  for i in range(n-1, -1, -1):
    B[i] = tf.matmul(tf.transpose(W[i+1]), B[i+1], name="B"+str(i))

  # Create gradient update. Make copy of variables and split update into
  # two run calls. Using single set of variables will gives updates that 
  # occasionally produce wrong results/NaN's because of data race
  
  dW = [0]*(n+1)
  updates1 = [0]*(n+1)  # compute updated value into Wcopy
  updates2 = [0]*(n+1)  # copy value back into W
  Wcopy = [0]*(n+1)
  for i in range(n+1):
    Wi_name = "Wcopy"+str(i)
    Wi_shape = (fs[i+1], fs[i])
    Wi_init = tf.zeros(dtype=dtype, shape=Wi_shape, name=Wi_name+"_init")
    Wcopy[i] = tf.Variable(Wi_init, name=Wi_name, trainable=False)
    
    dW[i] = tf.matmul(B[i], tf.transpose(A[i]), name="dW"+str(i))


  del dW[0]  # get rid of W[0] update
  
  # construct flattened gradient update vector
  dWf = tf.concat([vec(grad) for grad in dW], axis=0)

  Wf_copy = tf.Variable(tf.zeros(dtype=dtype, shape=Wf.shape,
                                 name="Wf_copy_init"),
                        name="Wf_copy")
  train_op1 = Wf_copy.assign(Wf - lr*dWf)
  train_op2 = Wf.assign(Wf_copy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  expected_losses = np.loadtxt("data/natural_gradient_multilayer_losses_gd.csv")
  observed_losses = []
  
  for i in range(20):
    observed_losses.append(sess.run([loss])[0])
    sess.run(train_op1)
    sess.run(train_op2)

  check_equal(observed_losses, expected_losses)

  
# convention, X0 is numpy, X is Tensor
def fisher_test():
  """Test computation of empirical Fisher matrix."""
  
  tf.reset_default_graph()

  tf.reset_default_graph()

  # load data into TF
  XY0 = np.genfromtxt('data/natural_gradient_multilayer_XY0.csv',
                      delimiter= ",")
  fs = np.genfromtxt('data/natural_gradient_multilayer_fs.csv',
                     delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers
  
  X0 = XY0[:-1,:]  # 2 x d
  Y0 = XY0[-1:,:]  # 1 x d
  dsize = X0.shape[1]
  Y_ = tf.placeholder(dtype, shape=Y0.shape, name="Y_holder")
  Y = tf.Variable(Y_, trainable=False)
  init_dict={Y_: Y0}

  W0f = v2c_np(np.genfromtxt('data/natural_gradient_multilayer_W0f.csv',
                             delimiter= ","))
  W0s = unflatten_np(W0f, fs[1:])  # Wf doesn't have first layer (data matrix)
  W0s.insert(0, X0)

  
  # initialize data + layers
  # W[0] is input matrix (X), W[n] is last matrix
  # A[1] has activations for W[1], equal to W[0]=X
  # A[n+1] has predictions
  W = [0]*(n+1)   # list of "W" matrices. 
  A = [0]*(n+2)
  A[0] = Identity(dsize)
  for i in range(n+1):
    # fs is off by 2 from common notation, ie W[0] has shape f[0],f[-1]
    Wi_name = "W"+str(i)
    Wi_shape = (fs[i+1], fs[i])
    Wi_holder = tf.placeholder(dtype, shape=Wi_shape, name=Wi_name+"_h")
    W[i] = tf.Variable(Wi_holder, name=Wi_name, trainable=(i>0))
    A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))
    init_dict[Wi_holder] = W0s[i]

  assert len(A) == n+2
  
  assert W[0].shape == (2, 10)
  assert W[1].shape == (2, 2)
  assert W[2].shape == (2, 2)
  assert W[3].shape == (1, 2)

  assert X0.shape == (2, 10)
  assert W0s[1].shape == (2, 2)
  assert W0s[2].shape == (2, 2)
  assert W0s[3].shape == (1, 2)
  
  assert A[0].shape == (10, 10)
  assert A[1].shape == (2, 10)
  assert A[2].shape == (2, 10)
  assert A[3].shape == (2, 10)
  assert A[4].shape == (1, 10)

  
  # input dimensions match
  assert W[0].get_shape() == X0.shape
  # output dimensions match
  assert W[-1].get_shape()[0], W[0].get_shape()[1] == Y0.shape
  assert A[n+1].get_shape() == Y0.shape

  err = Y - A[n+1]
  #  loss = (1./(2*dsize))*(err @ tf.transpose(err))
  loss = tf.reduce_sum(tf.square(err))/(2*dsize)
  lr = tf.Variable(0.5, dtype=dtype)
  
  # create backprop matrices
  # B[i] has backprop for matrix i
  B = [0]*(n+1)
  B[n] = -err/dsize
  for i in range(n-1, -1, -1):
    B[i] = tf.matmul(tf.transpose(W[i+1]), B[i+1], name="B"+str(i))

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  expected_losses = np.loadtxt("data/natural_gradient_multilayer_losses_regular.csv")

  expected_fisher = np.loadtxt("data/natural_gradient_multilayer_fisher0.csv",
                               delimiter= ",")

  block11 = khatri_rao(A[1], B[1]) @ tf.transpose(khatri_rao(A[1], B[1]))/dsize
  expected_block11 = expected_fisher[:4, :4]
  check_equal(sess.run(block11), expected_block11)

  # construct fisher matrix out of individual mats
  grads = tf.concat([khatri_rao(A[i], B[i]) for i in range(1, n+1)], axis=0)
  fisher = grads @ tf.transpose(grads) / dsize
  check_equal(sess.run(fisher), expected_fisher)

  # create vectorized parameter vector
  
  
  # Create gradient update. Make copy of variables and split update into
  # two run calls. Using single set of variables will gives updates that 
  # occasionally produce wrong results/NaN's because of data race
  
  dW = [0]*(n+1)
  updates1 = [0]*(n+1)  # compute updated value into Wcopy
  updates2 = [0]*(n+1)  # copy value back into W
  Wcopy = [0]*(n+1)
  for i in range(n+1):
    Wi_name = "Wcopy"+str(i)
    Wi_shape = (fs[i+1], fs[i])
    Wi_init = tf.zeros(dtype=dtype, shape=Wi_shape, name=Wi_name+"_init")
    Wcopy[i] = tf.Variable(Wi_init, name=Wi_name, trainable=False)
    
    dW[i] = tf.matmul(B[i], tf.transpose(A[i]), name="dW"+str(i))
    updates1[i] = Wcopy[i].assign(W[i]-lr*dW[i])
    updates2[i] = W[i].assign(Wcopy[i])

  del updates1[0]  # don't update input matrices
  del updates2[0]
  
  train_op1 = tf.group(*updates1)
  train_op2 = tf.group(*updates2)

  observed_losses = []
  for i in range(20):
    observed_losses.append(sess.run([loss])[0])
    sess.run(train_op1)
    sess.run(train_op2)

  check_equal(observed_losses, expected_losses)

# convention, X0 is numpy, X is Tensor
def natural_gradient_test():
  """Train network using empirical natural gradient."""
  
  tf.reset_default_graph()

  # load data into TF
  XY0 = np.genfromtxt('data/natural_gradient_multilayer_XY0.csv',
                      delimiter= ",")
  
  # should be 10,2,2,2
  fs = np.genfromtxt('data/natural_gradient_multilayer_fs.csv',
                     delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers
  
  X0 = XY0[:-1,:]  # 2 x d
  Y0 = XY0[-1:,:]  # 1 x d
  dsize = X0.shape[1]
  Y_ = tf.placeholder(dtype, shape=Y0.shape, name="Y_holder")
  Y = tf.Variable(Y_, trainable=False)
  init_dict={Y_: Y0}

  W0f = v2c_np(np.genfromtxt('data/natural_gradient_multilayer_W0f.csv',
                             delimiter= ","))
  W0s = unflatten_np(W0f, fs[1:])  # Wf doesn't have first layer (data matrix)
  W0s.insert(0, X0)

  
  # initialize data + layers
  # W[0] is input matrix (X), W[n] is last matrix
  # A[1] has activations for W[1], equal to W[0]=X
  # A[n+1] has predictions
  W = [0]*(n+1)   # list of "W" matrices. 
  A = [0]*(n+2)
  A[0] = Identity(dsize)

  dims = [(fs[i+1],fs[i]) for i in range(len(fs)-1)]
  sizes = [s[0]*s[1] for s in dims]
  Wf_size = np.sum(sizes[1:])
  Wf_holder = tf.placeholder(dtype, shape=W0f.shape)
  Wf = tf.Variable(Wf_holder, name="Wf")
  assert Wf.shape == W0f.shape
  init_dict[Wf_holder] = W0f
  
  W = unflatten(Wf, fs[1:])
  W.insert(0, tf.constant(X0))
  assert W[0].shape == [2, 10]
  assert W[1].shape == [2, 2]
  assert W[2].shape == [2, 2]
  assert W[3].shape == [1, 2]
                   
  for i in range(n+1):
    # fs is off by 2 from common notation, ie W[0] has shape f[0],f[-1]
    A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))
    
  assert len(A) == n+2
  assert A[0].shape == (10, 10)
  assert A[1].shape == (2, 10)
  assert A[2].shape == (2, 10)
  assert A[3].shape == (2, 10)
  assert A[4].shape == (1, 10)

  
  # input dimensions match
  assert W[0].get_shape() == X0.shape
  # output dimensions match
  assert W[-1].get_shape()[0], W[0].get_shape()[1] == Y0.shape
  assert A[n+1].get_shape() == Y0.shape

  err = Y - A[n+1]
  loss = tf.reduce_sum(tf.square(err))/(2*dsize)
  lr = tf.Variable(0.001, dtype=dtype)
  
  # create backprop matrices
  # B[i] has backprop for matrix i
  B = [0]*(n+1)
  B[n] = -err/dsize
  for i in range(n-1, -1, -1):
    B[i] = tf.matmul(tf.transpose(W[i+1]), B[i+1], name="B"+str(i))

  # Create gradient update. Make copy of variables and split update into
  # two run calls. Using single set of variables will gives updates that 
  # occasionally produce wrong results/NaN's because of data race
  
  dW = [0]*(n+1)
  updates1 = [0]*(n+1)  # compute updated value into Wcopy
  updates2 = [0]*(n+1)  # copy value back into W
  Wcopy = [0]*(n+1)
  for i in range(n+1):
    Wi_name = "Wcopy"+str(i)
    Wi_shape = (fs[i+1], fs[i])
    Wi_init = tf.zeros(dtype=dtype, shape=Wi_shape, name=Wi_name+"_init")
    Wcopy[i] = tf.Variable(Wi_init, name=Wi_name, trainable=False)
    
    dW[i] = tf.matmul(B[i], tf.transpose(A[i]), name="dW"+str(i))


  del dW[0]  # get rid of W[0] update
  
  # construct flattened gradient update vector
  dWf = tf.concat([vec(grad) for grad in dW], axis=0)


  # inverse fisher preconditioner
  grads = tf.concat([khatri_rao(A[i], B[i]) for i in range(1, n+1)], axis=0)
  fisher = grads @ tf.transpose(grads) / dsize
  ifisher = pseudo_inverse(fisher)

  Wf_copy = tf.Variable(tf.zeros(dtype=dtype, shape=Wf.shape,
                                 name="Wf_copy_init"),
                        name="Wf_copy")
  new_val_matrix = Wf - lr*(ifisher @ dWf)
  train_op1 = Wf_copy.assign(new_val_matrix)
  train_op2 = Wf.assign(Wf_copy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)

  # from notebook {0.347015, 0.301344, 0.260196, 0.224903, 0.193672...
  expected_losses = np.loadtxt("data/natural_gradient_multilayer_losses_fisher.csv")
  
  observed_losses = []
  for i in range(20):
    observed_losses.append(sess.run([loss])[0])
    sess.run(train_op1)
    sess.run(train_op2)

  check_equal(observed_losses, expected_losses)


def newton_test():
  """Test Newton-Rhapson implementation """
  
  tf.reset_default_graph()

  # load data into TF
  XY0 = np.genfromtxt('data/natural_gradient_multilayer_XY0.csv',
                      delimiter= ",")
  
  fs = np.genfromtxt('data/natural_gradient_multilayer_fs.csv',
                     delimiter= ",").astype(np.int32)
  def f(i): return fs[i+1]
  
  n = len(fs)-2    # number of layers
  
  X0 = XY0[:-1,:]  # 2 x d
  Y0 = XY0[-1:,:]  # 1 x d
  dsize = X0.shape[1]
  Y_ = tf.placeholder(dtype, shape=Y0.shape, name="Y_holder")
  Y = tf.Variable(Y_, trainable=False)
  init_dict={Y_: Y0}

  W0f = v2c_np(np.genfromtxt('data/natural_gradient_multilayer_W0f.csv',
                             delimiter= ","))
  W0s = unflatten_np(W0f, fs[1:])  # Wf doesn't have first layer (data matrix)
  W0s.insert(0, X0)

  
  # initialize data + layers
  # W[0] is input matrix (X), W[n] is last matrix
  # A[1] has activations for W[1], equal to W[0]=X
  # A[n+1] has predictions
  W = [0]*(n+1)   # list of "W" matrices. 
  A = [0]*(n+2)
  A[0] = Identity(dsize)

  dims = [(fs[i+1],fs[i]) for i in range(len(fs)-1)]
  sizes = [s[0]*s[1] for s in dims]
  Wf_size = np.sum(sizes[1:])
  Wf_holder = tf.placeholder(dtype, shape=W0f.shape)
  Wf = tf.Variable(Wf_holder, name="Wf")
  assert Wf.shape == W0f.shape
  init_dict[Wf_holder] = W0f
  
  W = unflatten(Wf, fs[1:])
  W.insert(0, tf.constant(X0))
  assert W[0].shape == [2, 10]
  assert W[1].shape == [2, 2]
  assert W[2].shape == [2, 2]
  assert W[3].shape == [1, 2]
                   
  for i in range(n+1):
    # fs is off by 2 from common notation, ie W[0] has shape f[0],f[-1]
    A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))
    
  assert len(A) == n+2
  assert A[0].shape == (10, 10)
  assert A[1].shape == (2, 10)
  assert A[2].shape == (2, 10)
  assert A[3].shape == (2, 10)
  assert A[4].shape == (1, 10)

  
  # input dimensions match
  assert W[0].get_shape() == X0.shape
  # output dimensions match
  assert W[-1].get_shape()[0], W[0].get_shape()[1] == Y0.shape
  assert A[n+1].get_shape() == Y0.shape

  err = Y - A[n+1]
  #  loss = (1./(2*dsize))*(err @ tf.transpose(err))
  loss = tf.reduce_sum(tf.square(err))/(2*dsize)
  lr = tf.Variable(0.001, dtype=dtype)
  
  # create backprop matrices
  # B[i] has backprop for matrix i
  B = [0]*(n+1)
  b = [0]*(n+1)  # like backprop matrix but no error
  B[n] = -err/dsize
  b[n] = Identity(fs[-1])
  for i in range(n-1, -1, -1):
    B[i] = t(W[i+1]) @ B[i+1]
    b[i] = t(W[i+1]) @ b[i+1]


  # off-diagonal products
  U = [list(range(n+1)) for _ in range(n+1)]
  for bottom in range(n+1):
    for top in range(n+1):
      prod = Identity(fs[top+1])
      for i in range(top, bottom-1, -1):
        prod = prod @ W[i]
      U[bottom][top] = prod

  # block i, j gives hessian block between layer i and layer j
  block = [list(range(n+1)) for _ in range(n+1)]
  for i in range(1, n+1):
    for j in range(1, n+1):
      if i == j:
        block[i][j] = kronecker(A[i]@t(A[i]), b[i]@t(b[i]))/dsize
      elif i < j:
        block[i][j] = (kr(A[i]@t(A[j]), b[i]@t(b[j])) -
                       kr((A[i]@t(B[j])), U[i+1][j-1]) @ Kmat(f(j),f(j-1)))
      else:
        block[i][j] = (kr(A[i]@t(A[j]), b[i]@t(b[j])) -
                       kr(t(U[j+1][i-1]), B[i]@t(A[j])) @ Kmat(f(j),f(j-1)))
        
  # remove leftmost blocks (those are with respect to W[0] which is input)
  del block[0]
  for row in block:
    del row[0]
    
  hess = concat_blocks(block)
  # Create gradient update. Make copy of variables and split update into
  # two run calls. Using single set of variables will gives updates that 
  # occasionally produce wrong results/NaN's because of data race
  
  dW = [0]*(n+1)
  updates1 = [0]*(n+1)  # compute updated value into Wcopy
  updates2 = [0]*(n+1)  # copy value back into W
  Wcopy = [0]*(n+1)
  for i in range(n+1):
    Wi_name = "Wcopy"+str(i)
    Wi_shape = (fs[i+1], fs[i])
    Wi_init = tf.zeros(dtype=dtype, shape=Wi_shape, name=Wi_name+"_init")
    Wcopy[i] = tf.Variable(Wi_init, name=Wi_name, trainable=False)
    
    dW[i] = tf.matmul(B[i], tf.transpose(A[i]), name="dW"+str(i))


  del dW[0]  # get rid of W[0] update
  
  # construct flattened gradient update vector
  dWf = tf.concat([vec(grad) for grad in dW], axis=0)


  # inverse fisher preconditioner
  grads = tf.concat([khatri_rao(A[i], B[i]) for i in range(1, n+1)], axis=0)
  fisher = grads @ tf.transpose(grads) / dsize
  # outer_sum[i] for i in range(1, n+1)
  ifisher = pseudo_inverse(fisher)

  Wf_copy = tf.Variable(tf.zeros(dtype=dtype, shape=Wf.shape,
                                 name="Wf_copy_init"),
                        name="Wf_copy")
  new_val_matrix = Wf - lr*(ifisher @ dWf)
  train_op1 = Wf_copy.assign(new_val_matrix)
  train_op2 = Wf.assign(Wf_copy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  expected_losses = np.loadtxt("data/natural_gradient_multilayer_losses_newton.csv")

  expected_hess = np.loadtxt("data/natural_gradient_multilayer_hess0.csv",
                             delimiter= ",")
  hess0 = sess.run(hess)
  np.savetxt("data/natural_gradient_multilayer_hess0_tf.csv", hess0,
             fmt="%.5f", delimiter=',')
  check_equal(expected_hess, sess.run(hess))

  
  observed_losses = []
  for i in range(20):
    observed_losses.append(sess.run([loss])[0])
    sess.run(train_op1)
    sess.run(train_op2)

  check_equal(observed_losses, expected_losses)


def relu_manual_vectorized_test():
  """Train network, with manual backprop, in vectorized form"""
  
  tf.reset_default_graph()

  # load data into TF
  XY0 = np.genfromtxt('data/natural_gradient_multilayer_XY0.csv',
                      delimiter= ",")
  
  fs = np.genfromtxt('data/natural_gradient_multilayer_fs.csv',
                     delimiter= ",").astype(np.int32)
  n = len(fs)-2    # number of layers
  
  X0 = XY0[:-1,:]  # 2 x d
  Y0 = XY0[-1:,:]  # 1 x d
  dsize = X0.shape[1]
  Y_ = tf.placeholder(dtype, shape=Y0.shape, name="Y_holder")
  Y = tf.Variable(Y_, trainable=False)
  init_dict={Y_: Y0}

  W0f = v2c_np(np.genfromtxt('data/natural_gradient_multilayer_W0f.csv',
                             delimiter= ","))
  W0s = unflatten_np(W0f, fs[1:])  # Wf doesn't have first layer (data matrix)
  W0s.insert(0, X0)

  
  # initialize data + layers
  # W[0] is input matrix (X), W[n] is last matrix
  # A[1] has activations for W[1], equal to W[0]=X
  # A[n+1] has predictions
  W = [0]*(n+1)   # list of "W" matrices. 
  A = [0]*(n+2)
  A[0] = Identity(dsize)

  dims = [(fs[i+1],fs[i]) for i in range(len(fs)-1)]
  sizes = [s[0]*s[1] for s in dims]
  Wf_size = np.sum(sizes[1:])
  Wf_holder = tf.placeholder(dtype, shape=W0f.shape)
  Wf = tf.Variable(Wf_holder, name="Wf")
  assert Wf.shape == W0f.shape
  init_dict[Wf_holder] = W0f
  
  W = unflatten(Wf, fs[1:])
  W.insert(0, tf.constant(X0))
  assert W[0].shape == [2, 10]
  assert W[1].shape == [2, 2]
  assert W[2].shape == [2, 2]
  assert W[3].shape == [1, 2]
                   
  for i in range(n+1):
    # fs is off by 2 from common notation, ie W[0] has shape f[0],f[-1]
    if i == 0:
      A[i+1] = tf.matmul(W[i], A[i], name="A"+str(i+1))
    else:
      A[i+1] = tf.nn.relu(tf.matmul(W[i], A[i], name="A"+str(i+1)))
    
  assert len(A) == n+2
  assert A[0].shape == (10, 10)
  assert A[1].shape == (2, 10)
  assert A[2].shape == (2, 10)
  assert A[3].shape == (2, 10)
  assert A[4].shape == (1, 10)

  
  # input dimensions match
  assert W[0].get_shape() == X0.shape
  # output dimensions match
  assert W[-1].get_shape()[0], W[0].get_shape()[1] == Y0.shape
  assert A[n+1].get_shape() == Y0.shape

  err = Y - A[n+1] 
  #  loss = (1./(2*dsize))*(err @ tf.transpose(err))
  loss = tf.reduce_sum(tf.square(err))/(2*dsize)
  lr = tf.Variable(0.5, dtype=dtype)
  
  # create backprop matrices
  # B[i] has backprop for matrix i
  B = [0]*(n+1)
  B[n] = (-err/dsize)*relu_mask(A[n+1])
  for i in range(n-1, -1, -1):
    B[i] = tf.matmul(tf.transpose(W[i+1]), B[i+1], name="B"+str(i))
    if i == 0:
      pass
    else:
      B[i] = B[i]*relu_mask(A[i+1])

  # Create gradient update. Make copy of variables and split update into
  # two run calls. Using single set of variables will gives updates that 
  # occasionally produce wrong results/NaN's because of data race
  
  dW = [0]*(n+1)
  updates1 = [0]*(n+1)  # compute updated value into Wcopy
  updates2 = [0]*(n+1)  # copy value back into W
  Wcopy = [0]*(n+1)
  for i in range(n+1):
    Wi_name = "Wcopy"+str(i)
    Wi_shape = (fs[i+1], fs[i])
    Wi_init = tf.zeros(dtype=dtype, shape=Wi_shape, name=Wi_name+"_init")
    Wcopy[i] = tf.Variable(Wi_init, name=Wi_name, trainable=False)
    
    dW[i] = tf.matmul(B[i], tf.transpose(A[i]), name="dW"+str(i))


  del dW[0]  # get rid of W[0] update
  
  # construct flattened gradient update vector
  dWf = tf.concat([vec(grad) for grad in dW], axis=0)

  Wf_copy = tf.Variable(tf.zeros(dtype=dtype, shape=Wf.shape,
                                 name="Wf_copy_init"),
                        name="Wf_copy")
  train_op1 = Wf_copy.assign(Wf - lr*dWf)
  train_op2 = Wf.assign(Wf_copy)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer(), feed_dict=init_dict)
  
  expected_losses = np.loadtxt("data/relu_losses_regular.csv")
  # from relus.nb
  #  {0.539407, 0.256027, 0.25684, 0.248212, 0.247842, 0.244276, 0.243793...

  
  observed_losses = []
  for i in range(20):
    observed_losses.append(sess.run([loss])[0])
    sess.run(train_op1)
    sess.run(train_op2)

  np.testing.assert_allclose(observed_losses, expected_losses)
  check_equal(observed_losses, expected_losses)


if __name__ == '__main__':
  fisher_test()
  gd_test()
  gd_manual_test()
  gd_manual_vectorized_test()
  natural_gradient_test()
  relu_manual_vectorized_test()
  print("%s tests passed" %(sys.argv[0]))
  #  newton_test()
