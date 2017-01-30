# mac: tf -- 90 usec,
# tf low-level with fetches -- 44 usec
# tf low-level -- 36 usec
# numpy -- 20 usec
# 
# xeon: tf -- 130 usec, tf low level -- 77, tf low level+XLA -- 20 usec, numpy -- 30
#
# benchmark tf
# python tiny_tf.py
#
# benchmark tf using low level API
# python tiny_tf.py fast
#
# benchmark tf using low level API
# python tiny_tf.py fast-nofetch
#
# bencharmk tf using low level API + XLA
# python tiny_tf.py fastxla
#
# bencharmk tf using low level API + XLA
# python tiny_tf.py fastxla-nofetch
#
# benchmark numpy
# python tiny_tf.py np


import os, sys
os.environ["CUDA_VISIBLE_DEVICES"]=""

from tensorflow.python import pywrap_tensorflow
from tensorflow.python import pywrap_tensorflow as tf_session
from tensorflow.python.framework import errors
import tensorflow as tf
import numpy as np
import time

try:
    from tensorflow.contrib.compiler import jit
    jit_scope = jit.experimental_jit_scope
except:
    print("No XLA for you")
    pass

if len(sys.argv)>1 and sys.argv[1]=='np':
    run_numpy = True
else:
    run_numpy = False

if len(sys.argv)>1 and 'fast' in sys.argv[1]:
    run_fast = True
else:
    run_fast = False
    
if len(sys.argv)>1 and 'xla' in sys.argv[1]:
    run_fastxla = True
else:
    run_fastxla = False
    
n = 600

def create_graph():
    def _create_graph():
        a = tf.constant(np.random.random((n, 64)).astype(np.float32))
        b = tf.constant(np.random.random((64, 64)).astype(np.float32))
        c = tf.constant(np.random.random((64, 64)).astype(np.float32))
        x = tf.random_uniform((1, n))

        y = tf.matmul(x, a)
        y = tf.matmul(y, b)
        y = tf.matmul(y, c)
        return y

    if run_fastxla:
        with jit_scope(compile_ops=True):
            return _create_graph()
    else:
        return _create_graph()
        
config = tf.ConfigProto()
                        
sess = tf.Session(config=config)

a_n = np.empty((n, 64))
b_n = np.empty((64, 64))
c_n = np.empty((64, 64))
dtype = np.float32
def f_numpy(): return np.empty((1, n), dtype=dtype).dot(a_n).dot(b_n).dot(c_n)

# setup low level args for TF_Run call
session = sess._session
options=None
feed_dict = {}

# uncomment lines below if you want to fetch things
fetch_list = [b'MatMul_2:0']
target_list = []

if len(sys.argv)>1 and 'nofetch' in sys.argv[1]:
    fetch_list=[]
    target_list=[b'MatMul_2']
    
run_metadata = None
status_orig = errors.raise_exception_on_not_ok_status()
status = pywrap_tensorflow.TF_NewStatus()

def fast_tf():
    return tf_session.TF_Run(session, options,
                             feed_dict, fetch_list, target_list,
                             status, run_metadata)

num_iters = 5000
warmup_iters = 2
iter_times = np.zeros((num_iters+warmup_iters,))
y = create_graph()
for i in range(num_iters+warmup_iters):
    iter_start = time.time()
    if i == warmup_iters:
        start_time = time.time()
    if run_numpy:
        f_numpy()
    elif (run_fast or run_fastxla) and i>=warmup_iters:
        fast_tf()
    else:
        sess.run(y)

    iter_end = time.time()
    iter_times[i] = iter_end - iter_start

#import pdb; pdb.set_trace()    
    
end_time = time.time()
print("Per iteration min: %d, avg: %d"%(min(iter_times)*10**6, (time.time()-start_time)/num_iters*10**6))
