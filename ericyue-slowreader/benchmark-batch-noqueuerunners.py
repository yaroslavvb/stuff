# range size is  1000000
# range queue 900000, batch queue 100000, 81510.55 per second
# range d 900000, batch d 100000
# range queue 800000, batch queue 200000, 71304.15 per second
# range d -100000, batch d 100000
# range queue 700000, batch queue 300000, 65481.16 per second
#
# When using enqueue_many=False
# range size is  1000000
# range queue 999000, batch queue 1000, 11485.93 per second
# range d 999000, batch d 1000
# range queue 998000, batch queue 2000, 13163.80 per second
# range d -1000, batch d 1000
# ange queue 997000, batch queue 3000, 13048.20 per second

import tensorflow as tf
import time, os, sys
from tensorflow.python.client import timeline

dump_timeline = False
enqueue_many  = True
enqueue_many_size = 1000
steps_to_validate = 200
epoch_number = 2
thread_number = 2
batch_size = 100

capacity = 2*10**6
# don't use too high of limit, 10**9 hangs (overflows to negative in TF?)
a_queue = tf.train.range_input_producer(limit=10**3, num_epochs=2000,
                                        capacity=capacity, shuffle=False)

# manually run the queue runner for a bit
config = tf.ConfigProto(log_device_placement=False)
config.operation_timeout_in_ms=5000   # terminate on long hangs
sess = tf.InteractiveSession("", config=config)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


a_queue_qr = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)[0]
for i in range(1000):
    sess.run(a_queue_qr.enqueue_ops)


# check the sizes
range_size_node = "input_producer/fraction_of_2000000_full/fraction_of_2000000_full_Size:0"

# size gives raw size rather than number of batches
batch_size_node = "batch/fifo_queue_Size:0"

print("range size is ", sess.run(range_size_node))

# now create batch and run it manually
# use size of 2 or get TypeError: 'Tensor' object is not iterable.
# (possibly singleton list get auto-packed into a single Tensor)
if enqueue_many:
    a_batch = []
    for i in range(enqueue_many_size):
        a_batch.append(a_queue.dequeue())
        
    b_batch = tf.train.batch([a_batch], batch_size=batch_size,
                             capacity=capacity, enqueue_many=enqueue_many)
        
else:
    [b, _] = tf.train.batch([a_queue.dequeue()]*2, batch_size=batch_size,
                                capacity=capacity, enqueue_many=enqueue_many)


start_time = time.time()
old_range_size, old_batch_size = (0, 0)

batch_qr = [qr for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS) if qr.name.startswith("batch")][0]

for i in range(10):
    for i in range(100): # put some elements on queue
        sess.run(batch_qr.enqueue_ops)

    if dump_timeline == True:
        run_metadata = tf.RunMetadata()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

        sess.run(batch_qr.enqueue_ops, run_metadata=run_metadata,
                 options=run_options)
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        trace_fn = '/tmp/ericyue/process.timeline.json'
        graph_fn = '/tmp/ericyue/process.graph.pbtxt'
        metadata_fn = '/tmp/ericyue/process.metadata.pbtxt'
        open(trace_fn, 'w').write(trace.generate_chrome_trace_format())
        open(graph_fn, 'w').write(str(tf.get_default_graph().as_graph_def()))
        open(metadata_fn, 'w').write(str(run_metadata))

        sys.exit()

    
    new_range_size, new_batch_size = sess.run([range_size_node, batch_size_node])
    
    new_time = time.time()
    rate = (new_batch_size-old_batch_size)/(new_time-start_time)
    print("range queue %d, batch queue %d, %.2f per second"%(new_range_size,
                                                             new_batch_size,
                                                             rate))
    print("range d %d, batch d %d" %(new_range_size - old_range_size,
                                     new_batch_size - old_batch_size))
    start_time = time.time()
    old_range_size, old_batch_size = new_range_size, new_batch_size 


