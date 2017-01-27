# qr on 4096 x 4096
# tf 6.89
# np openblas 11.38
# np mkl: 2.36

import tensorflow as tf
import time
import numpy as np

np.__config__.show()

try:
  tf.reset_default_graph()
  n = 2048*2
  mat = tf.Variable(tf.random_uniform((n,n)))
  qr = tf.qr(mat)
  sess = tf.Session(config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0))))
  sess.run(tf.initialize_all_variables())
  sess.run(qr[0].op)
  start_time = time.time()
  sess.run(qr[0].op)
  end_time = time.time()
  print("TF QR on %d by %d matrix in %.2f seconds"%(n, n, end_time-start_time))
except:
  print("No tf")

a = np.random.randn(n, n)
start_time = time.time()
q, r = np.linalg.qr(a)
end_time = time.time()
print("numpy QR on %d by %d matrix in %.2f seconds"%(n, n, end_time-start_time))
