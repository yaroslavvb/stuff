'''
Running on I7/GTX 1080

MKL version b'Intel(R) Math Kernel Library Version 2017.0.3 Product Build 20170413 for Intel(R) 64 architecture applications'
TF version:  b'v1.3.0-rc1-3233-g07bf1d3'
TF url:  https://github.com/tensorflow/tensorflow/commit/07bf1d3
PyTorch version 0.2.0_4
Timing in ms for 1534 x 1534 SVD of type <class 'numpy.float32'>
numpy default        min:   328.32, median:   328.76, mean:   343.25
numpy gesvd          min:  1424.12, median:  1425.25, mean:  1447.09
numpy gesdd          min:   243.74, median:   243.98, mean:   244.40
TF CPU               min:   965.73, median:  1118.91, mean:  1089.15
TF GPU               min:  5525.59, median:  5726.54, mean:  5987.72
PyTorch CPU          min:  1241.66, median:  1372.49, mean:  1389.59
PyTorch GPU          min:   450.84, median:   455.17, mean:   471.32
'''

from scipy import linalg  # for svd
import numpy as np
import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"  # nospam

import tensorflow as tf
import gc; gc.disable()
import torch

NUM_RUNS = 11
dtype = np.float32
N=1534


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

timeline_counter = 0
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
def traced_run(fetches):
    """Runs fetches, dumps timeline files in current directory."""
    
    from tensorflow.python.client import timeline

    global timeline_counter
    run_metadata = tf.RunMetadata()

    results = sess.run(fetches,
                       options=run_options,
                       run_metadata=run_metadata);
    tl = timeline.Timeline(step_stats=run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format(show_memory=True,
                                          show_dataflow=False)
    open("timeline_%d.json"%(timeline_counter,), "w").write(ctf)
    open("stepstats_%d.pbtxt"%(timeline_counter,), "w").write(str(
        run_metadata.step_stats))
    timeline_counter+=1
    return results

def benchmark(message, func):
    time_list = []
    for i in range(NUM_RUNS):
        start_time = time.perf_counter()
        func()
        time_list.append(time.perf_counter()-start_time)

    time_list = 1000*np.array(time_list)  # get seconds, convert to ms
    if len(time_list)>0:
        min = np.min(time_list)
        median = np.median(time_list)
        formatted = ["%.2f"%(d,) for d in time_list[:10]]
        result = "min: %8.2f, median: %8.2f, mean: %8.2f"%(min, median, np.mean(time_list))
    else:
        result = "empty"
    print("%-20s %s"%(message, result))
    

if np.__config__.get_info("lapack_mkl_info"):
    print("MKL version", get_mkl_version())
else:
    print("no MKL")

print("TF version: ", tf.__git_version__)
print("TF url: ", get_tensorflow_version_url())
print("PyTorch version", torch.version.__version__)


with tf.device("/cpu:0"):
    data = tf.random_uniform((N, N), dtype=dtype)
    tf_svd_cpu = tf.group(*tf.svd(data))

with tf.device("/gpu:0"):
    data = tf.random_uniform((N, N), dtype=dtype)
    tf_svd_gpu = tf.group(*tf.svd(data))

np_data = np.random.random((N, N)).astype(dtype)
print("Timing in ms for %d x %d SVD of type %s"%(N, N, dtype))
def func(): linalg.svd(np_data)
benchmark("numpy default", func)

def func(): linalg.svd(np_data, lapack_driver='gesvd');
benchmark("numpy gesvd", func)

def func(): linalg.svd(np_data, lapack_driver='gesdd');
benchmark("numpy gesdd", func)

sess = tf.Session()

def func(): sess.run(tf_svd_cpu)
benchmark("TF CPU", func)

def func(): sess.run(tf_svd_gpu)
benchmark("TF GPU", func)

import torch
def func(): torch.svd(torch.rand((N,N)))
benchmark("PyTorch CPU", func)
def func(): torch.svd(torch.rand((N,N)).cuda())
benchmark("PyTorch GPU", func)
