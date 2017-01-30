# Example of catching GPU OOM error
# http://stackoverflow.com/questions/41942538/tensorflow-gpu-memory-error-try-except-not-catching-the-error

import tensorflow as tf

try:
    with tf.device("gpu:0"):
        a = tf.Variable(tf.ones((10000, 10000)))
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
except:
    print("Caught error")
    import pdb; pdb.set_trace()
