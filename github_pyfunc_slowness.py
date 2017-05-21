# Example of py_func slowing down future computations
# On Mac
# time 1 0.007195033016614616
# time 2 0.0070790809113532305
# time 3 0.008019614033401012
#
# On Xeon V3:
# time 1 0.011401358991861343
# time 2 0.011637557297945023
# time 3 0.012380894273519516
#
# On Mac without MKL installed:
# time 1 0.011707969009876251
# time 2 0.011970046092756093
# time 3 0.011933871079236269

import numpy as np
import scipy
import scipy.linalg
import tensorflow as tf
import timeit
sess = tf.Session()
a = np.random.random((300, 300))
a = a.dot(a.T)
best_time = np.inf
for i in range(10):
    s = timeit.default_timer()
    scipy.linalg.eigh(a)
    e = timeit.default_timer()
    if e - s < best_time:
        best_time = e - s
print("time 1", best_time)
       
np.linalg.svd(np.random.randn(2, 300))
 
best_time = np.inf
for i in range(10):
    s = timeit.default_timer()
    scipy.linalg.eigh(a)
    e = timeit.default_timer()
    if e - s < best_time:
        best_time = e - s
print("time 2", best_time)

ret = tf.py_func(np.linalg.svd, [np.random.randn(2, 300)], [tf.float64, tf.float64, tf.float64])
sess.run(ret)

best_time = np.inf
for i in range(10):
    s = timeit.default_timer()
    scipy.linalg.eigh(a)
    e = timeit.default_timer()
    if e - s < best_time:
        best_time = e - s
print("time 3", best_time)
