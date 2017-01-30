# from http://stackoverflow.com/questions/41920371/tensorflow-multi-threaded-queuerunner?noredirect=1#comment71036438_41920371

import tensorflow as tf
import numpy as np

batch_size = 10
iters = 100
a = tf.train.range_input_producer(10, shuffle=False).dequeue()
b = tf.train.range_input_producer(10, shuffle=False).dequeue()
c1, c2 = tf.train.batch([a,b], num_threads=batch_size, batch_size=batch_size)
config = tf.ConfigProto()
config.operation_timeout_in_ms=5000   # terminate on long hangs

sess = tf.InteractiveSession(config=config)
sess.run([tf.initialize_all_variables()])

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord)

results = []
for i in range(iters):
    d1 = sess.run(tf.reduce_all(tf.equal(c1, c2)))
    results.append(d1)
print("mismatches: %d/%d"%(iters-sum(results), iters))

coord.request_stop()
coord.join(threads)
