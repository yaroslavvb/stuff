# from https://github.com/tensorflow/tensorflow/issues/7251
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from tensorflow.python.client.timeline import Timeline

with tf.device("/gpu:0"):
    x = tf.ones(100, name="x")
    idxs = tf.range(100)

    for i in range(10):
        y = tf.identity(x, name="identity-"+str(i))
        x = tf.dynamic_stitch([idxs, idxs], [x, y], name="stitch-"+str(i))

config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
sess = tf.InteractiveSession(config=config)
metadata = tf.RunMetadata()
sess.run(x, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,
                                  output_partition_graphs=True),
         run_metadata=metadata)

timeline = Timeline(metadata.step_stats)
with open("dynamic_stitch_gpu_profile.json", "w") as f:
    f.write(timeline.generate_chrome_trace_format())
with open("dynamic_stitch_gpu_profile.pbtxt", "w") as f:
    f.write(str(metadata))
