# test whether memory gets cleared on creating new sessions
import sys, os, math, random


import tensorflow as tf
import numpy as np

if __name__=='__main__':
  try:
    sess = tf.Session()
    size = 12000
    num_runs = 10

    images = tf.random_uniform([size, size])
    var = tf.Variable(tf.ones_like(images))
    sess.run(var.initializer)
    for i in range(10):
      def relu(x):
        return tf.where(tf.less(x, 0.0), x, x, name='leaky_relu')
      cost = tf.reduce_sum(relu(images+var))

      grads = tf.gradients(cost, var)
      _, memuse = sess.run([grads, tf.contrib.memory_stats.MaxBytesInUse()])
      print("Run %d, GBs in use %.2f"%(i, memuse/10**9))
  except:
    pass
  finally:
    [memuse] = sess.run([tf.contrib.memory_stats.MaxBytesInUse()])
    print("Memory GBs in use %.2f"%(memuse/10**9,))
    

#    576000000
# 2017-09-21 14:53:23.483412: I tensorflow/core/framework/log_memory.cc:35] __LOG_MEMORY__ MemoryLogTensorOutput { step_id: 2 kernel_name: "gradients/leaky_relu_grad/zeros_like" tensor { dtype: DT_FLOAT shape { dim { size: 144000000 } } allocation_description { requested_bytes: 576000000 allocated_bytes: 576000000 allocator_name: "GPU_0_bfc" allocation_id: 6 ptr: 1109438113536 } } }
