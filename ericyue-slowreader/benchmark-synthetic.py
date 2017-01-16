# [1484610732] time[  0.17] step[      2000] speed[1173998]
# [1484610733] time[  0.14] step[      4000] speed[1462395]
# [1484610733] time[  0.14] step[      6000] speed[1473741]
# [1484610733] time[  0.14] step[      8000] speed[1468095]

import tensorflow as tf
import time

# try benchmarking

# don't use too high of limit, 10**9 hangs (overflows to negative in TF?)
a_queue = tf.train.range_input_producer(limit=10**3, capacity=1000)
#a_queue = tf.train.string_input_producer(["hello"])
a = a_queue.dequeue()

steps_to_validate = 2000
epoch_number = 2
thread_number = 2
batch_size = 100

config = tf.ConfigProto(log_device_placement=True)
config.operation_timeout_in_ms=5000   # terminate on long hangs
sess = tf.InteractiveSession("", config=config)

tf.train.start_queue_runners()

step = 0
start_time = time.time()
while True:
    step+=1
    sess.run(a.op)
    if step % steps_to_validate == 0:
        end_time = time.time()
        sec = (end_time - start_time)
        print("[{}] time[{:6.2f}] step[{:10d}] speed[{:6d}]".format(
            str(end_time).split(".")[0],sec, step,
            int((steps_to_validate*batch_size)/sec)
        ))
        start_time = end_time
