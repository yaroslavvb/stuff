import sys
import tensorflow as tf

cluster_spec = tf.train.ClusterSpec({
  "a": { 0: "localhost:8000" },
  "b": { 0: "localhost:8001" },
})

jobname = "a"
taskid = 0
server = tf.train.Server(cluster_spec, jobname, taskid)

with tf.device("/job:a/task:0/cpu:0"):
  queue = tf.FIFOQueue(
    capacity=100, dtypes=[tf.int64],
    shapes=[[]], shared_name="a_queue", name="a_queue")

if jobname == "a" and taskid == 0:
  enqueue_op = queue.enqueue(10)
  sess = tf.Session(server.target)
  while True:
    sess.run(enqueue_op)
else:
  dequeue_op = queue.dequeue()
  sess = tf.Session(server.target)
  while True:
    print(sess.run(dequeue_op))
