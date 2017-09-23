'''
Created on Sep 21, 2017

modified from voldemaro

On I7/GTX 1080

MKL detected
MKL version b'Intel(R) Math Kernel Library Version 2017.0.3 Product Build 20170413 for Intel(R) 64 architecture applications'
TF version:  b'v1.3.0-rc1-2487-g088cdea'
TF url:  https://github.com/tensorflow/tensorflow/commit/088cdea
pre-warming: 9.510
TF GPU 5.553296
TF CPU 6.508888
numpy default: 0.302414
numpy gesdd: 0.251199
numpy gesvd 1.485335

'''
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
import time;
import numpy.linalg as NLA;
from scipy import linalg  # for svd

def get_tensorflow_version_url():
    import tensorflow as tf
    version=tf.__version__
    commit = tf.__git_version__
    # commit looks like this
    # 'v1.0.0-65-g4763edf-dirty'
    commit = commit.replace("'","")
    if commit.endswith('-dirty'):
        dirty = True
        commit = commit[:-len('-dirty')]
    commit=commit.rsplit('-g', 1)[1]
    url = 'https://github.com/tensorflow/tensorflow/commit/'+commit
    return url

def get_mkl_version():
    import ctypes
    import numpy as np
    ver = np.zeros(199, dtype=np.uint8)
    mkl = ctypes.cdll.LoadLibrary("libmkl_rt.so")
    mkl.MKL_Get_Version_String(ver.ctypes.data_as(ctypes.c_char_p), 198)
    return ver[ver != 0].tostring()


if np.__config__.get_info("lapack_mkl_info"):
    print("MKL detected")
    print("MKL version", get_mkl_version())
else:
    print("no MKL")

print("TF version: ", tf.__git_version__)
print("TF url: ", get_tensorflow_version_url())

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


sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

# Initialize all tensorflow variables
start = time.time();
sess.run(specVarGPU.initializer, feed_dict={init_holder_gpu: svd_array});
sess.run(specVarCPU.initializer, feed_dict={init_holder_cpu: svd_array});
    
[d, e1, e2]  = sess.run([D2_gpu.op, E1_gpu.op,  E2_gpu.op]);
[d, e1, e2]  = sess.run([D2_cpu.op, E1_cpu.op,  E2_cpu.op]);
u, s, v = linalg.svd(svd_array);
u, s, v = linalg.svd(svd_array, lapack_driver='gesdd');
u, s, v = linalg.svd(svd_array, lapack_driver='gesvd');
print('pre-warming: %.3f'%(time.time()-start))

start_time = time.time();
[d, e1, e2]  = sess.run([D2_gpu.op, E1_gpu.op,  E2_gpu.op]);
print("TF GPU %.6f"%(time.time() - start_time))

[d, e1, e2]  = sess.run([D2_cpu.op, E1_cpu.op,  E2_cpu.op]);
print("TF CPU %.6f" %(time.time() - start_time))

# Defaut numpy (gesdd)
start = time.time();
u, s, v = linalg.svd(svd_array);
print('numpy default: %.6f'%(time.time() - start))

start = time.time();
u, s, v = linalg.svd(svd_array);
print('numpy gesdd: %.6f'%(time.time() - start))

# Numpy gesvd
start = time.time();
u, s, v = linalg.svd(svd_array, lapack_driver='gesvd');
print('numpy gesvd %.6f'%(time.time() - start))

