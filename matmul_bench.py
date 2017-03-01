import tensorflow as tf

import numpy
import time

n=4096*2
tf.reset_default_graph()
X = tf.Variable(tf.random_uniform((n, n)))
y = tf.matmul(X, X)
sess = tf.Session()
sess.run(X.initializer)
sess.run(y.op)
start_time = time.time()
sess.run(y.op)
elapsed = time.time() - start_time
num_ops = n**2*(n-1) + n**3
print("--- %s seconds tensorflow---" % (elapsed))
print("%.2f Tops/sec"%(num_ops/elapsed/1000**4))

