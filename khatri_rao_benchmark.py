# 0.6 - 0.8 for 10x10 khatri rao
# After improvement: 0.02 seconds
import tensorflow as tf
import util as u
import time
import os
import sys


def benchmark_construct(dims, iters, dtype):
  A = tf.ones((dims, dims), dtype=dtype)
  B = tf.ones((dims, dims), dtype=dtype)
  prods = []
  time0 = time.time()
  for i in range(iters):
    prods.append(u.khatri_rao(A,B))
  elapsed = time.time() - time0
  print("Constructed %d x %d kr %d times in %.2f seconds"%(A.shape[0], B.shape[0], iters, elapsed))
  
def benchmark_execute(dims, iters, dtype):
  A = tf.random_uniform((dims, dims), dtype=dtype)
  B = tf.random_uniform((dims, dims), dtype=dtype)
  prods = []
  for i in range(iters):
    prods.append(u.khatri_rao(A,B))
  elapsed_times = []
  sess = tf.Session()
  elapsed_times = []
  u.reset_time()
  for i in range(10):
    time0 = time.time()
    sess.run(tf.group(*prods))
    elapsed_times.append(time.time()-time0)
    u.record_time()
  u.summarize_time()


if __name__ == '__main__':
  dims = 10
  iters = 10
  dtype = tf.float32
  benchmark_construct(dims, iters, dtype)
  benchmark_execute(dims, iters, dtype)
  
