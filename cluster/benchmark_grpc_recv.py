#!/usr/bin/env python
#
# Dependencies:
# portpicker (pip install portpicker)
# tcmalloc4 (sudo apt-get install google-perftools)
#
# TODO: add baseline numbers
# Generating profile:
#
# rm /tmp/profile*
# python benchmark_grpc_recv.py --data_mb=512 --profile
# export p=/tmp/profile.out.0_27680
# google-pprof `which python` $p --svg > /tmp/profile.0.svg
# export p=/tmp/profile.out.1_27683
# google-pprof `which python` $p --svg > /tmp/profile.1.svg


import os
import portpicker
import subprocess
import sys
import tensorflow as tf
import threading
import time

flags = tf.flags
flags.DEFINE_integer("iters", 1000, "number of times to repeat experiment")
flags.DEFINE_integer("iters_per_step", 100, "number of additions per step")
flags.DEFINE_integer("data_mb", 128, "size of vector in MBs")
flags.DEFINE_boolean("verbose", False, "whether to have verbose logging")
flags.DEFINE_boolean("profile", False, "whether to collect CPU profile")

# internal flags, set by client
flags.DEFINE_string("task_index", "", "# of current task")
flags.DEFINE_string("port0", "12222", "port of worker1, used as master")
flags.DEFINE_string("port1", "12223", "port of worker2")
FLAGS = flags.FLAGS


flags.DEFINE_string('localdir_prefix', '/temp/logs',
                     'where to mirror worker logs locally')
flags.DEFINE_string('logdir_prefix', '/efs/logs',
                     'where to dump EFS logs')
flags.DEFINE_string('name', 'default',
                    'tag used to keep track of machines in this experiment')


# setup local cluster from flags

def session_config():
  optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)
  graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
  config = tf.ConfigProto(graph_options=graph_options,
                          intra_op_parallelism_threads=10,
                          inter_op_parallelism_threads=10)


host = "127.0.0.1"
def clusterspec():
  cluster = {"worker": [host+":"+FLAGS.port0, host+":"+FLAGS.port1]}
  return tf.train.ClusterSpec(cluster).as_cluster_def()
  
  
def create_graph(device0, device1):
  """Create graph that keeps var1 on device0, var2 on device1 and adds them"""
  
  tf.reset_default_graph()
  dtype=tf.int32
  params_size = 250*1000*FLAGS.data_mb # 1MB is 250k integers

  with tf.device(device0):
    var1 = tf.get_variable("var1", [params_size], dtype,
                             initializer=tf.ones_initializer())
  with tf.device(device1):
    var2 = tf.get_variable("var2", [params_size], dtype,
                           initializer=tf.ones_initializer())
    add_op = var1.assign_add(var2)

  init_op = tf.global_variables_initializer()
  return init_op, add_op

def create_done_queue(i):
  """Queue used to signal death for i'th worker."""
  
  with tf.device("/job:worker/task:%s" % (i)):
    return tf.FIFOQueue(1, tf.int32, shared_name="done_queue"+
                        str(i))

from tensorflow.python.summary import summary as summary_lib
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.util import compat
from tensorflow.core.util import event_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.python.training import training_util # TOOD: not needed?


def make_event(tag, value, step):
  event = event_pb2.Event(
      wall_time=time.time(),
      step=step,
      summary=summary_pb2.Summary(
          value=[summary_pb2.Summary.Value(
              tag=tag, simple_value=value)]))
  return event

def run_benchmark(sess, init_op, add_op):
  """Returns MB/s rate of addition."""


  logdir=FLAGS.logdir_prefix+'/'+FLAGS.name
  os.system('mkdir -p '+logdir)
  
  # TODO: make events follow same format as eager writer
  writer = pywrap_tensorflow.EventsWriter(compat.as_bytes(logdir+'/events'))
  filename = compat.as_text(writer.FileName())
  training_util.get_or_create_global_step()

  sess.run(init_op)

  for step in range(FLAGS.iters):
    start_time = time.time()
    for i in range(FLAGS.iters_per_step):
      sess.run(add_op.op)

    elapsed_time = time.time() - start_time
    rate = float(FLAGS.iters)*FLAGS.data_mb/elapsed_time
    event = make_event('rate', rate, step)
    writer.WriteEvent(event)
    writer.Flush()
  writer.Close()
  # add event


def run_benchmark_local():
  ops = create_graph(None, None)
  sess = tf.Session(config=session_config())
  return run_benchmark(sess, *ops)


def run_benchmark_distributed():
  ops = create_graph("/job:worker/task:0", "/job:worker/task:1")
  queues = [create_done_queue(0), create_done_queue(1)]

  # launch distributed service


  port0, port1 = [portpicker.pick_unused_port() for _ in range(2)]
  flags = " ".join(sys.argv)  # pass parent flags to children
  
  def run_worker(w):
    my_env = os.environ.copy()
    if not FLAGS.verbose:
      my_env["CUDA_VISIBLE_DEVICES"] = ""
      my_env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    if FLAGS.profile:
      my_env["LD_PRELOAD"]="/usr/lib/libtcmalloc_and_profiler.so.4"
      my_env["CPUPROFILE"]="/tmp/profile.out.%s"%(w)
    cmd = "python %s --task=%d --port0=%s --port1=%s"%(flags, w, port0, port1)
    subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT,
                     env=my_env)
    
  run_worker(0)
  run_worker(1)

  sess = tf.Session("grpc://%s:%s"%(host, port0), config=session_config())
  rate = run_benchmark(sess, *ops)

  # bring down workers
  if FLAGS.verbose:
    print("Killing workers.")
  sess.run(queues[1].enqueue(1))
  # todo: sleep to avoid killing master too early?
  sess.run(queues[0].enqueue(1))  # bring down master last
  
  return rate

if __name__=='__main__':
  if not FLAGS.task_index:

    rate1 = run_benchmark_local()
    rate2 = run_benchmark_distributed()

    if FLAGS.verbose:
      print("Adding data in %d MB chunks" %(FLAGS.data_mb))
    print("Local rate:       %.2f MB/s" %(rate1,))
    print("Distributed rate: %.2f MB/s" %(rate2,))

  else: # Launch TensorFlow server
    server = tf.train.Server(clusterspec(), config=session_config(),
                             job_name="worker",
                             task_index=int(FLAGS.task_index))
    queue = create_done_queue(FLAGS.task_index)
    sess = tf.Session(server.target, config=session_config())
    sess.run(queue.dequeue())
    time.sleep(1) # give chance for master session.run call to return
    if FLAGS.verbose:
      print("Worker %s quitting." %(FLAGS.task_index))
