# On Titan X (Pascal)
# 8192 x 8192 matmul took: 0.10 sec, 11304.59 G ops/sec
# http://stackoverflow.com/questions/41804380/testing-gpu-with-tensorflow-matrix-multiplication
#
# On V100/fp16
# 4096 x 4096 matmul took: 0.00191 sec, 72111.44 G ops/sec
# 8192 x 8192 matmul took: 0.01 sec, 76310.61 G ops/sec
# 8192 x 8192 matmul took: 0.16 sec  54970.06 G ops/sec
# 10000 x 10000 matmul took: 0.03 sec, 65290.49 G ops/sec

from __future__ import print_function

import ctypes
import errno
from ctypes.util import find_library
from functools import partial

CLOCK_PROCESS_CPUTIME_ID = 2  # time.h
CLOCK_MONOTONIC_RAW = 4
clockid_t = ctypes.c_int
time_t = ctypes.c_long
class timespec(ctypes.Structure):
    _fields_ = [
        ('tv_sec', time_t),         # seconds
        ('tv_nsec', ctypes.c_long)  # nanoseconds
    ]
_clock_gettime = ctypes.CDLL(find_library('rt'), use_errno=True).clock_gettime
_clock_gettime.argtypes = [clockid_t, ctypes.POINTER(timespec)]


def clock_gettime(clk_id):
    tp = timespec()
    if _clock_gettime(clk_id, ctypes.byref(tp)) < 0:
        err = ctypes.get_errno()
        msg = errno.errorcode[err]
        if err == errno.EINVAL:
            msg += (" The clk_id specified is not supported on this system"
                    " clk_id=%r") % (clk_id,)
        raise OSError(err, msg)
    return tp.tv_sec + tp.tv_nsec * 1e-9

try:
    from time import perf_counter, process_time
except ImportError:  # Python <3.3
    perf_counter = partial(clock_gettime, CLOCK_MONOTONIC_RAW)
    perf_counter.__name__ = 'perf_counter'
    process_time = partial(clock_gettime, CLOCK_PROCESS_CPUTIME_ID)
    process_time.__name__ = 'process_time'


import math
import os
import sys
import numpy as np
import tensorflow as tf
import time

import argparse
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

parser.add_argument('--dtype', type=str, default='float32',
                    help='dtype, float32 or float16')
parser.add_argument('--agg', type=str, default='min',
                    help='min, mean or median')
args = parser.parse_args()

def bench(n):
  if args.dtype == 'float32':
    dtype = tf.float32
  elif args.dtype == 'float16':
    dtype = tf.float16
  else:
    assert False, 'unknown dtype '+args.dtype
  with tf.device("/gpu:0"):
    matrix1 = tf.Variable(tf.ones((n, n), dtype=dtype))
    matrix2 = tf.Variable(tf.ones((n, n), dtype=dtype))
    product = tf.matmul(matrix1, matrix2)


  config = tf.ConfigProto()
  sess = tf.Session(config=config)
  
  sess.run(tf.global_variables_initializer())
  iters = 11

  # pre-warming
  sess.run(product.op)

  times = []
  for i in range(iters):
    start = perf_counter()
    sess.run(product.op)
    times.append(perf_counter()-start)

  ops = n**3 + (n-1)*n**2 # n^2*(n-1) additions, n^3 multiplications

  times_ms = 1000*np.array(times)  # get seconds, convert to ms
  if len(times_ms)>0:
    min = np.min(times_ms)
    median = np.median(times_ms)
    formatted = ["%.2f"%(d,) for d in times_ms[:10]]
    #    print("Times: min: %.2f, median: %.2f, mean: %.2f"%(min, median,
    #                                                        np.mean(times_ms)))

  if args.agg == 'min':
    elapsed_ms = np.min(times_ms)
  elif args.agg == 'mean':
    elapsed_ms = np.mean(times_ms)
  elif args.agg == 'median':
    elapsed_ms = np.median(times_ms)
  else:
    assert False, 'unknown aggregation method: ' + args.agg
    
  rate = ops/elapsed_ms/10**9
  #  print('\n %d x %d matmul took: %.4f ms, %.2f G ops/sec' % (n, n,
  #                                                             elapsed_ms,
  #                                                             rate,))
  return rate

def main():
  steps = 8 # number of steps between n doubling

  np.set_printoptions(suppress=True)
  with open("times.csv", "w") as myfile:
    myfile.write("\n")
    
  for i in range(20*steps):
    n = int(math.pow(2, float(i)/steps))
    rate = bench(n)
    print("%d,%.10f" %(n, rate))
    with open("times.csv", "a") as myfile:
      myfile.write("%d,%.10f\n"%(n, rate))
  
if __name__=='__main__':
  main()
