import datetime
import pytz
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.client import timeline
import glob
import json
import time
import math
import numpy as np
import os
import tensorflow as tf

steps_to_validate = 20
epoch_number = 2
thread_number = 2
batch_size = 100
min_after_dequeue = 1000
capacity = thread_number * batch_size + min_after_dequeue

filename_queue = tf.train.string_input_producer(
      ["./data.zlib"],
      shuffle=True,
      seed = int(time.time()),
      num_epochs=epoch_number)

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader(options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB))
    _, serialized_example = reader.read(filename_queue)
    return serialized_example

serialized_example = read_and_decode(filename_queue)
batch_serialized_example = tf.train.shuffle_batch(
    [serialized_example],
    batch_size=batch_size,
    num_threads=thread_number,
    capacity=capacity,
    min_after_dequeue=min_after_dequeue)
features = tf.parse_example(
    batch_serialized_example,
    features={
        "label": tf.FixedLenFeature([], tf.float32),
        "ids": tf.VarLenFeature(tf.int64),
        "values": tf.VarLenFeature(tf.float32),
    })

batch_labels = features["label"]
batch_ids = features["ids"]
batch_values = features["values"]

init_op = tf.global_variables_initializer()

sess = tf.Session()
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()
sess.run(init_op,options=run_options, run_metadata=run_metadata)
sess.run(tf.local_variables_initializer(),options=run_options, run_metadata=run_metadata)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)
#time.sleep(10)

start_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
step = 1
try:
    while not coord.should_stop():
        f1,f2,f3 = sess.run([batch_ids,batch_values,batch_labels],options=run_options, run_metadata=run_metadata)
        step +=1
        if step % steps_to_validate == 0:
            end_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
            sec = (end_time - start_time).total_seconds()
            print("[{}] time[{:6.2f}] step[{:10d}] speed[{:6d}]".format(
                str(end_time).split(".")[0],sec, step,
                int((steps_to_validate*batch_size)/sec)
                ))
            start_time = end_time
        if step > 100:
          break


except tf.errors.OutOfRangeError:
    print("Done training after reading all data")
finally:
    coord.request_stop()
    print("coord stopped")

coord.join(threads)

tl = timeline.Timeline(run_metadata.step_stats)
ctf = tl.generate_chrome_trace_format()
with open('timeline.json', 'w') as f:
       f.write(ctf)
print("all done")
