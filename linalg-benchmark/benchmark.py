"""
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
"""

import scipy
from scipy import linalg  # for svd
import numpy as np
import os
import sys
import time
# import gc; gc.disable()
import torch
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"  # nospam

HAVE_GPU = tf.test.is_gpu_available()
NUM_RUNS = 11
dtype = np.float32
N=int(os.environ.get('linalg_benchmark_N', 1534))

def main():
  if np.__config__.get_info("lapack_mkl_info"):
    print("MKL version", get_mkl_version())
  else:
    print("not using MKL")

  print("TF version: ", tf.__git_version__, get_tensorflow_version_url())
  print("PyTorch version", torch.version.__version__)

  print("Scipy version: ", scipy.version.full_version)
  print("Numpy version: ", np.version.full_version)
  #  print("Python version: ", sys.version)
  print_cpu_info()

  np_data = np.random.random((N, N)).astype(dtype)
  print("Timing in ms for %d x %d SVD"%(N, N))

  def func(): linalg.svd(np_data)
  benchmark("numpy default", func)

  def func(): linalg.svd(np_data, lapack_driver='gesvd');
  benchmark("numpy gesvd", func)

  def func(): linalg.svd(np_data, lapack_driver='gesdd');
  benchmark("numpy gesdd", func)

  def func(): torch.svd(torch.rand((N,N)))
  benchmark("PyTorch CPU", func)
  def func(): torch.svd(torch.rand((N,N)).cuda())
  benchmark("PyTorch GPU", func)

  # do TensorFlow last because:
  # 1. it hogs all GPU memory by default
  # 2. if it runs out of GPU memory, instead of throwing exception it crashes
  #    the entire process with something like
  # F Check failed: cusolverDnCreate(&cusolver_dn_handle) == CUSOLVER_STATUS_SUCCESS Failed to create cuSolverDN instance.
  sess = tf.Session()
  # have to assign to variable, otherwise TF optimizes it out
  with tf.device("/cpu:0"):
    variable = tf.Variable(tf.zeros((N, N)), dtype=dtype)
    data = tf.random_uniform((N, N), dtype=dtype)
    # s/u/v convention https://github.com/tensorflow/tensorflow/pull/13850
    s, u, v = tf.svd(data)
    svd_assign = variable.assign(u)
  def func(): sess.run(svd_assign)
  benchmark("TF CPU", func)

  with tf.device("/gpu:0"):
    variable = tf.Variable(tf.zeros((N, N)), dtype=dtype)
    data = tf.random_uniform((N, N), dtype=dtype)
    s, u, v = tf.svd(data)
    svd_assign = variable.assign(u)
  def func(): sess.run(svd_assign)
  benchmark("TF GPU", func)



def get_tensorflow_version_url():
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

  # this recipe only works on Linux
  try:
    ver = np.zeros(199, dtype=np.uint8)
    mkl = ctypes.cdll.LoadLibrary("libmkl_rt.so")
    mkl.MKL_Get_Version_String(ver.ctypes.data_as(ctypes.c_char_p), 198)
    return ver[ver != 0].tostring()
  except:
    return 'unknown'


timeline_counter = 0
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
def traced_run(fetches):
    """Runs fetches, dumps timeline files in current directory."""
    
    from tensorflow.python.client import timeline
    global sess

    global timeline_counter
    run_metadata = tf.RunMetadata()

    results = sess.run(fetches,
                       options=run_options,
                       run_metadata=run_metadata)
    tl = timeline.Timeline(step_stats=run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format(show_memory=True,
                                          show_dataflow=False)
    open("timeline_%d.json"%(timeline_counter,), "w").write(ctf)
    open("stepstats_%d.pbtxt"%(timeline_counter,), "w").write(str(
        run_metadata.step_stats))
    timeline_counter+=1
    return results


def benchmark(message, func):
    if 'gpu' in message.lower() and not HAVE_GPU:
        print(f"{message:<20} no GPU detected")
        return
    time_list = []
    try:
      for i in range(NUM_RUNS):
          start_time = time.perf_counter()
          func()
          time_list.append(time.perf_counter()-start_time)

      time_list = 1000*np.array(time_list)  # get seconds, convert to ms
      if len(time_list)>0:
          min = np.min(time_list)
          median = np.median(time_list)
          formatted = ["%.2f"%(d,) for d in time_list[:10]]
          result = f"min: {min:8.2f}, median: {median:8.2f}, mean: {np.mean(time_list):8.2f}"
      else:
          result = "empty"
      print(f"{message:<20} {result}")
    except Exception as e:
      print(f"{message:<20} failed with {e}")

def print_cpu_info():
  try:
    for l in open("/proc/cpuinfo").read().split('\n'):
      if 'model name' in l:
        ver = l
        break
  except:
    ver = 'unknown'
    
  # core counts from https://stackoverflow.com/a/23378780/419116
  print("CPU version: ", ver)
  sys.stdout.write("CPU logical cores: ")
  sys.stdout.flush()
  os.system("echo $([ $(uname) = 'Darwin' ] && sysctl -n hw.logicalcpu_max || lscpu -p | egrep -v '^#' | wc -l)")
  sys.stdout.write("CPU physical cores: ")
  sys.stdout.flush()
  os.system("echo $([ $(uname) = 'Darwin' ] && sysctl -n hw.physicalcpu_max || lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l)")

  # get mapping of logical cores to physical sockets
  import re
  socket_re = re.compile(".*?processor.*?(?P<cpu>\d+).*?physical id.*?(?P<socket>\d+).*?power", flags=re.S)
  from collections import defaultdict
  socket_dict = defaultdict(list)
  try:
    for cpu, socket in socket_re.findall(open('/proc/cpuinfo').read()):
      socket_dict[socket].append(cpu)
  except:
    pass
  print("CPU physical sockets: ", len(socket_dict))


if __name__=='__main__':
  main()
