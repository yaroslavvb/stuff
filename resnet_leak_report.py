# test whether memory gets cleared on creating new sessions
import sys, os, math, random
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


import tensorflow as tf
import numpy as np

if __name__=='__main__':
  for i in range(10):
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    size = 12000
    example_queue = tf.FIFOQueue(1, dtypes=[tf.float32], shapes=[[size]])
    from tensorflow.python.ops import gen_random_ops
    image = tf.random_uniform([size])
    example_enqueue_op = example_queue.enqueue([image])
    sess.run(example_enqueue_op)
    sess.run(example_queue.close())

    images = example_queue.dequeue_many(1)
    images = tf.concat([images]*size, axis=0)
    var = tf.Variable(tf.ones_like(images))

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    def relu(x):
      return tf.where(tf.less(x, 0.0), x, x, name='leaky_relu')
    cost = tf.reduce_sum(relu(images+var))

    grads = tf.gradients(cost, var)
    _, memuse = sess.run([grads, tf.contrib.memory_stats.MaxBytesInUse()])
    print("Run %d, GBs in use %.1f"%(i, memuse/10**9))

    sess.close()
    del sess
