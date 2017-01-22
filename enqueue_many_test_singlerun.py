# Test multiple enqueue many in single .run call
import os, sys
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]=""
import tensorflow as tf

def create_session():
    config = tf.ConfigProto(log_device_placement=False)
    config.operation_timeout_in_ms=5000   # terminate on long hangs
    config.gpu_options.per_process_gpu_memory_fraction=0.3 # don't hog all vRAM
    sess = tf.InteractiveSession("", config=config)
    return sess

import time
import threading
import os
os.environ['PYTHONUNBUFFERED'] = 'True'


from google.protobuf.internal import api_implementation
assert api_implementation._default_implementation_type == 'cpp'


from tensorflow.python.client import timeline
tf.reset_default_graph()

reverse = False
if len(sys.argv)>1:
    assert sys.argv[1] == 'reverse'
    reverse = True
    
n = 10**6
dtype = tf.int32
queue = tf.FIFOQueue(capacity=2*n, dtypes=[dtype], shapes=[()])
zeros = tf.Variable(tf.zeros((n), name="0", dtype=dtype))
ones = tf.Variable(tf.ones((n), name="1", dtype=dtype))
enqueue_zeros = queue.enqueue_many(zeros, name="zeros")
enqueue_ones = queue.enqueue_many(ones, name="ones")
sess = create_session()
sess.run(tf.global_variables_initializer())

op = tf.group(enqueue_zeros, enqueue_ones)

start_time0 = time.time()
run_metadatas = []
def run_op(op):
    start_time = time.time()
    print("%10.2f ms: starting op %s\n" % ((start_time-start_time0)*1000, op.name), flush=True, end='')
    
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess.run(op, options=options, run_metadata=run_metadata)
    end_time = time.time()
    print("%10.2f ms: ending op %s\n" % ((end_time-start_time0)*1000, op.name), flush=True, end='')
    run_metadatas.append(run_metadata)



threads = [threading.Thread(group=None, target=run_op, args=(op,))]
    
for t in threads:
    t.start()

# wait for threads to finish
for t in threads:
    t.join()

# generate merged timeline
merged_metadata = tf.RunMetadata()
for run_metadata in run_metadatas:
    merged_metadata.MergeFrom(run_metadata)

tl = timeline.Timeline(merged_metadata.step_stats)
ctf = tl.generate_chrome_trace_format()
with open(sys.argv[0]+'_timeline.json', 'w') as f:
    f.write(ctf)

assert sess.run(queue.size()) == 2*n
result = sess.run(queue.dequeue_many(2*n))
padding = np.array([0])

diffs = np.concatenate([padding, result])-np.concatenate([result, padding])
print("Interleaving detected: %s" % (abs(diffs).sum()>2))
