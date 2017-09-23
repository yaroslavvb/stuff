'''
Running on I7/GTX 1080

MKL version b'Intel(R) Math Kernel Library Version 2017.0.3 Product Build 20170413 for Intel(R) 64 architecture applications'
TF version:  b'v1.3.0-rc1-2487-g088cdea'
TF url:  https://github.com/tensorflow/tensorflow/commit/088cdea
Timing in ms for 1534 x 1534 SVD of type <class 'numpy.float32'>
numpy default        min:   243.52, median:   246.42, mean:   259.49
numpy gesvd          min:  1294.13, median:  1296.60, mean:  1298.93
numpy gesdd          min:   242.36, median:   242.80, mean:   245.32
TF CPU               min:   950.77, median:  1080.64, mean:  1050.79
TF GPU               min:  5483.26, median:  5520.19, mean:  5571.15
'''

from scipy import linalg  # for svd
import numpy as np
import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"  # nospam

import tensorflow as tf
import gc; gc.disable()

NUM_RUNS = 11
dtype = np.float32
N=1534;


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


svd_array = np.random.random_sample((N,N)).astype(dtype);

with tf.device("/gpu:0"):
    init_holder_gpu = tf.placeholder(dtype, shape=(N,N))
    specVarGPU = tf.random_uniform((N,N), dtype=dtype)
    [D2_gpu, E1_gpu,  E2_gpu] = tf.svd(specVarGPU);
with tf.device("/cpu:0"):
    init_holder_cpu = tf.placeholder(dtype, shape=(N,N))
    specVarCPU = tf.random_uniform((N,N), dtype=dtype)
    [D2_cpu, E1_cpu,  E2_cpu] = tf.svd(specVarCPU);

print("Timing in ms for %d x %d SVD of type %s"%(N, N, dtype))
def func(): linalg.svd(svd_array)
benchmark("numpy default", func)

def func(): linalg.svd(svd_array, lapack_driver='gesvd');
benchmark("numpy gesvd", func)

def func(): linalg.svd(svd_array, lapack_driver='gesdd');
benchmark("numpy gesdd", func)

sess = tf.Session()

def func(): sess.run([D2_cpu.op, E1_cpu.op,  E2_cpu.op])
benchmark("TF CPU", func)

def func(): sess.run([D2_gpu.op, E1_gpu.op,  E2_gpu.op])
benchmark("TF GPU", func)
