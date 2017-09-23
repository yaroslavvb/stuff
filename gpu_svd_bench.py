'''
Created on Sep 21, 2017

modified from voldemaro

Tensorflow GPU SVD ---: 5.9389612674713135 s
Tensorflow CPU SVD ---: 6.898533821105957 s
numpy default  ---: 0.42925524711608887 s
numpy gesvd  ---: 1.2659776210784912 s

'''
import numpy as np
import tensorflow as tf
import time;
import numpy.linalg as NLA;
from scipy import linalg  # for svd

# initializing variables: 1.79107093811 s
# Tensorflow SVD ---: 7.43185997009 s
# numpy default  ---: 2.80239009857 s
# numpy gesvd  ---: 117.758116961 s


N=1534;

dtype = np.float32
svd_array = np.random.random_sample((N,N)).astype(dtype);

with tf.device("/gpu:0"):
    init_holder_gpu = tf.placeholder(dtype, shape=(N,N))
    specVarGPU = tf.Variable(init_holder_gpu, dtype=dtype);
    [D2_gpu, E1_gpu,  E2_gpu] = tf.svd(specVarGPU);
with tf.device("/cpu:0"):
    init_holder_cpu = tf.placeholder(dtype, shape=(N,N))
    specVarCPU = tf.Variable(init_holder_cpu, dtype=dtype);
    [D2_cpu, E1_cpu,  E2_cpu] = tf.svd(specVarCPU);


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# Initialize all tensorflow variables
start = time.time();
sess.run(specVarGPU.initializer, feed_dict={init_holder_gpu: svd_array});
sess.run(specVarCPU.initializer, feed_dict={init_holder_cpu: svd_array});
print('initializing variables: {} s'.format(time.time()-start))
    
start_time = time.time();
[d, e1, e2]  = sess.run([D2_gpu.op, E1_gpu.op,  E2_gpu.op]);
print("Tensorflow GPU SVD ---: {} s" . format(time.time() - start_time));

[d, e1, e2]  = sess.run([D2_cpu.op, E1_cpu.op,  E2_cpu.op]);
print("Tensorflow CPU SVD ---: {} s" . format(time.time() - start_time));

# Defaut numpy (gesdd)
start = time.time();
u, s, v = linalg.svd(svd_array);
print('numpy default  ---: {} s'.format(time.time() - start));

# Numpy gesvd
start = time.time();
u, s, v = linalg.svd(svd_array, lapack_driver='gesvd');
print('numpy gesvd  ---: {} s'.format(time.time() - start));

