import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline 
 

sess = tf.Session()
a = tf.placeholder(tf.float32)
b = a*2
c0 = sess.run([b], feed_dict={a:2.})

run_metadata = tf.RunMetadata()
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_options.output_partition_graphs=True

c0 = sess.run([b], feed_dict={a:2.}, options=run_options,
              run_metadata=run_metadata)
with open("feed_dict.pbtxt", "w") as f:
    f.write(str(run_metadata))
