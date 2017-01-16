# [1484609202] time[  0.01] step[        20] speed[360350]
# [1484609202] time[  0.00] step[        40] speed[1129322]
# [1484609202] time[  0.00] step[        60] speed[546168]
# [1484609202] time[  0.00] step[        80] speed[709696]
# [1484609202] time[  0.00] step[       100] speed[1112399]
# [1484609202] time[  0.00] step[       120] speed[1506033]

import tensorflow as tf
import time

filename_queue = tf.train.string_input_producer(["./data.zlib"],
                                                shuffle=False,
                                                seed = int(time.time()))

reader = tf.TFRecordReader(options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB))
_, serialized_example = reader.read(filename_queue)

reader = tf.TFRecordReader(options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB))
_, serialized_example = reader.read(filename_queue)

sess = tf.InteractiveSession()
tf.train.start_queue_runners()

batch_size = 100
steps_to_validate = 20

step = 0
start_time = time.time()
while True:
    step+=1
    sess.run(serialized_example.op)
    if step % steps_to_validate == 0:
        end_time = time.time()
        sec = (end_time - start_time)
        print("[{}] time[{:6.2f}] step[{:10d}] speed[{:6d}]".format(
            str(end_time).split(".")[0],sec, step,
            int((steps_to_validate*batch_size)/sec)
        ))
        start_time = end_time
