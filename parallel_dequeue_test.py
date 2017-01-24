# 0.12.1
# v0.12.0-10-g4d924e7-dirty
# [array([1, 2, 3, 4, 0], dtype=int32), True]
# [array([8, 6, 7, 9, 5], dtype=int32), True]
# [array([11, 12, 13, 14, 10], dtype=int32), True]
# [array([16, 17, 18, 19, 15], dtype=int32), True]
#
# In HEAD (from Jan 17)
# 0.12.head
# 0.12.1-1878-g76d5960-dirty
# [array([0, 0, 0, 0, 0], dtype=int32), False]
# [array([1, 1, 1, 1, 1], dtype=int32), False]
# [array([2, 2, 2, 2, 2], dtype=int32), False]
# [array([3, 3, 3, 3, 3], dtype=int32), False]

import os, sys
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]=""
import tensorflow as tf

def create_session():
    config = tf.ConfigProto(log_device_placement=False)
    #    config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
    sess = tf.InteractiveSession("", config=config)
    return sess

import time
import threading
import os
os.environ['PYTHONUNBUFFERED'] = 'True'

n = 100
num_parallel = 5
dtype = tf.int32
queue = tf.FIFOQueue(capacity=n, dtypes=[dtype], shapes=[()])
enqueue_op = queue.enqueue_many(tf.range(n))

dequeue_ops = []
for i in range(num_parallel):
    dequeue_ops.append(queue.dequeue())

if hasattr(tf, "stack"):
    batch = tf.stack(dequeue_ops)
else:
    batch = tf.pack(dequeue_ops)
all_unique = tf.equal(tf.size(tf.unique(batch)[0]), num_parallel)
sess = create_session()
sess.run(enqueue_op)
print(tf.__version__)
print(tf.__git_version__)
for i in range(n//num_parallel):
    print(sess.run([batch, all_unique]))
