# from http://stackoverflow.com/questions/41920371/tensorflow-multi-threaded-queuerunner?noredirect=1#comment71036438_41920371

import tensorflow as tf
import numpy as np
import time

batch_size = 4
iters = 100
a = tf.train.range_input_producer(10, shuffle=False, name="a", capacity=batch_size*iters).dequeue()
b = tf.train.range_input_producer(10, shuffle=False, name="b", capacity=batch_size*iters).dequeue()
c1, c2 = tf.train.batch([a,b], num_threads=batch_size, batch_size=batch_size, capacity=iters)
config = tf.ConfigProto()
config.operation_timeout_in_ms=5000   # terminate on long hangs
#import pdb; pdb.set_trace()
sess = tf.InteractiveSession(config=config)
sess.run([tf.initialize_all_variables()])

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord)


time.sleep(1)
coord.request_stop()
coord.join(threads)
#print("Queue runners: ")
#for qr in tf.get_default_graph().get_collection(tf.GraphKeys.QUEUE_RUNNERS):
#    print("name: %s" %(qr.name))
#    print("queue_name: %s" %(qr.queue.name))
#    print("number of enqueue ops: %d"%(len(qr.enqueue_ops),))

results = []
for i in range(iters):
    d1,list1,list2 = sess.run([tf.reduce_all(tf.equal(c1, c2)), c1, c2])
    if not d1:
        print(list1)
        print(list2)
    results.append(d1)
print("mismatches: %d/%d"%(iters-sum(results), iters))


coord.request_stop()
