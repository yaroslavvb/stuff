# [1484611992] time[  0.00] step[       420] speed[613695]
# [1484611992] time[  0.00] step[       440] speed[501141]
# [1484611992] time[  0.01] step[       460] speed[351428]
# [1484611992] time[  0.00] step[       480] speed[450032]
# [1484611993] time[  0.14] step[       500] speed[ 14419]
# [1484611993] time[  0.15] step[       520] speed[ 13662]
# [1484611993] time[  0.14] step[       540] speed[ 13960]
# [1484611993] time[  0.15] step[       560] speed[ 13069]

import tensorflow as tf
import time


steps_to_validate = 20
epoch_number = 2
thread_number = 2
batch_size = 100

capacity = 2*10**6
# don't use too high of limit, 10**9 hangs (overflows to negative in TF?)
a_queue = tf.train.range_input_producer(limit=10**3, capacity=capacity)

# use size of 2 or get TypeError: 'Tensor' object is not iterable.
# (possibly singleton list get auto-packed into a single Tensor)
[b, _] = tf.train.batch([a_queue.dequeue()]*2, batch_size=100,
                        capacity=capacity)


config = tf.ConfigProto(log_device_placement=True)
config.operation_timeout_in_ms=5000   # terminate on long hangs
sess = tf.InteractiveSession("", config=config)

tf.train.start_queue_runners()
time.sleep(5)

step = 0
start_time = time.time()
while True:
    step+=1
    sess.run(b.op)
    if step % steps_to_validate == 0:
        end_time = time.time()
        sec = (end_time - start_time)
        print("[{}] time[{:6.2f}] step[{:10d}] speed[{:6d}]".format(
            str(end_time).split(".")[0],sec, step,
            int((steps_to_validate*batch_size)/sec)
        ))
        start_time = end_time
