# Troubleshooting
# https://github.com/tensorflow/tensorflow/issues/13433#issuecomment-351722017

import tensorflow as tf
import numpy as np

def sessrun(*args, **kwargs):
  """Helper to do sess.run and save run_metadata"""
  global sess, run_metadata
  
  run_metadata = tf.RunMetadata()

  kwargs['options'] = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  kwargs['run_metadata'] = run_metadata
  result = sess.run(*args, **kwargs)
  first_entry = args[0]
  # have to do this because sess.run(tensor) is same as sess.run([tensor]) 
  if isinstance(first_entry, list):
    if len(first_entry) == 0 and len(args) == 1:
      return None
    first_entry = first_entry[0]

import urllib.request
response = urllib.request.urlopen("https://raw.githubusercontent.com/yaroslavvb/chain_constant_memory/master/mem_util.py")
open("mem_util.py", "wb").write(response.read())

import mem_util


dtype = tf.float32
dtype_size = 4 # bytes
#shape = (1000,1000*1000)
shape = (100, 1000*1000)
total_size = np.prod(shape)*dtype_size
print("Variable with %.1f GB" %(total_size/1e9,))
w = tf.Variable(tf.random_uniform(shape,dtype=dtype),dtype=dtype)
sess = tf.Session()
sessrun(tf.global_variables_initializer())
print(sess.run(w[0,0]))

mem_util.print_memory_timeline(run_metadata)
