import sys
import tensorflow as tf

cluster_spec = tf.train.ClusterSpec({
  "a": { 0: "localhost:8000" },
  "b": { 1: "localhost:8001" },
})

DOFAIL=True

jobname = "b"
taskid = 1
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
  with tf.device("/job:b/task:1"):
    out = queue.dequeue()
    queue_b = tf.FIFOQueue(capacity=100, dtypes=[tf.int64], shapes=[[]], name="b_queue")
    if DOFAIL:
        out = tf.cond(tf.equal(out, 10), lambda: queue_b.enqueue(out), lambda: tf.no_op())
        g = tf.get_default_graph()
        from tensorflow.core.framework import attr_value_pb2
        op = g.get_operation_by_name('cond/b_queue_enqueue/Switch_1')
        op.node_def.attr["_class"].CopyFrom(attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(s=[])))

        op = g.get_operation_by_name('cond/b_queue_enqueue/Switch')
        op.node_def.attr["_class"].CopyFrom(attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(s=[])))
        
        with open('fail.pbtxt', 'w') as outf:
            outf.write(str(tf.get_default_graph().as_graph_def()))
    else:
        enq = queue_b.enqueue(out)
        no_op = tf.no_op()
        out = tf.cond(tf.equal(out, 10), lambda: enq, lambda: no_op)
        with open('pass.pbtxt', 'w') as outf:
            outf.write(str(tf.get_default_graph().as_graph_def()))
        

  sess = tf.Session(server.target)
  while True:
    print(sess.run(out))
