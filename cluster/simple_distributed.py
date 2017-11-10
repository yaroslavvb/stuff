#!/usr/bin/env python
# Launches 

import gc
import os
import portpicker
import subprocess
import sys
import tensorflow as tf
import tempfile
import threading
import time
from tensorflow.python.training import training_util
from tensorflow.python.platform import gfile

from tensorflow.contrib.summary import summary_ops
from tensorflow.contrib.summary import gen_summary_ops


flags = tf.flags
flags.DEFINE_integer("iters", 100000, "Maximum number of additions")
flags.DEFINE_integer("warmup_iters", 1, "warmup iterations")
flags.DEFINE_integer("data_mb", 128, "size of vector in MBs")
flags.DEFINE_boolean("verbose", False, "extra logging")
flags.DEFINE_boolean("sanity_check", False, "run sanity check on results")
flags.DEFINE_boolean("profile", False, "whether to run distributed version and "
                                           "collect CPU profile")
flags.DEFINE_boolean("in_process", True, "do bencharmk on in-process master")
flags.DEFINE_string("direction", "t2p", "which direction to profile, either "
                    "tensorflow to python (t2p) or python to tensorflow (p2t)")
flags.DEFINE_string("logdir", os.environ["HOME"]+"/efs/client_transfer_benchmark", "location of tensorboard events")

# internal flags, set by client
flags.DEFINE_string("worker_type", "launcher", "launcher or client or worker")
flags.DEFINE_string("port", "12222", "port of master")
FLAGS = flags.FLAGS


host = "127.0.0.1"
def clusterspec():
  cluster = {"worker": [host+":"+FLAGS.port]}
  return tf.train.ClusterSpec(cluster).as_cluster_def()


def log(s):
  if FLAGS.verbose: print(s)


def session_config():
  optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)
  config = tf.ConfigProto(
    graph_options=tf.GraphOptions(optimizer_options=optimizer_options))
  config.log_device_placement = False
  config.allow_soft_placement = False
  return config


def launch_distributed_service():
  port = portpicker.pick_unused_port()
  
  def launch_worker(worker_type):
    my_env = os.environ.copy()
    if not FLAGS.verbose:
      my_env["CUDA_VISIBLE_DEVICES"] = ""
      my_env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if FLAGS.profile:
      my_env["LD_PRELOAD"]="/usr/lib/libtcmalloc_and_profiler.so.4"
      my_env["CPUPROFILE"]="/tmp/profile.out.%s"%(worker_type)

    args = ["python"] + sys.argv + ["--port="+str(port),
                                    "--worker_type="+worker_type]
    proc = subprocess.Popen(args, stderr=subprocess.STDOUT, env=my_env)
    log("worker %s pid %s"%(worker_type, proc.pid))
    
  launch_worker("worker")
  launch_worker("client")

def run_benchmark(master, direction=None):
  """Connect to master and run simple TF->Python transfer benchmark."""


  from tensorflow.python.summary import summary as summary_lib
  from tensorflow.python import pywrap_tensorflow
  from tensorflow.python.util import compat
  from tensorflow.core.util import event_pb2
  from tensorflow.core.framework import summary_pb2


  def make_event(tag, value, step):
    event = event_pb2.Event(
        wall_time=time.time(),
        step=step,
        summary=summary_pb2.Summary(
            value=[summary_pb2.Summary.Value(
                tag=tag, simple_value=value)]))
    return event
    
  if not direction:
    os.system('mkdir -p '+FLAGS.logdir)

    # todo: unique filenames like with contrib.summary writer
    writer = pywrap_tensorflow.EventsWriter(compat.as_bytes(FLAGS.logdir+'/events'))
    filename = compat.as_text(writer.FileName())

    training_util.get_or_create_global_step()
    sess = tf.InteractiveSession()
    step = 0
    while True:
      p_to_t = run_benchmark(master, 'p->t')
      print("recoridng", p_to_t, "to", FLAGS.logdir)
      t_to_p = run_benchmark(master, 't->p')
      
      event = make_event('p->t', p_to_t, step)
      writer.WriteEvent(event)
      event = make_event('t->p', t_to_p, step)
      writer.WriteEvent(event)
      writer.Flush()
      step+=1
      
    writer.Close()
    return
  
  assert FLAGS.warmup_iters > 0
  gc.disable()
  
  dtype = tf.int32
  params_size = 250*1000*FLAGS.data_mb # 1MB is 250k integers
#  params = tf.get_variable("params", [params_size], dtype,
#                           initializer=tf.ones_initializer())
  params = tf.Variable(tf.ones([params_size], dtype=dtype), name='params')
  params_read = params.read_value()   # prevent caching
  params_holder = tf.placeholder(dtype)
  params_write = params.assign(params_holder)
  done_queue = create_done_queue(0)
  init_op = tf.global_variables_initializer()
  sess = tf.Session(master, config=session_config())
  sess.run(init_op)
  result = sess.run(params_read)
  
  total = 0
  for i in range(FLAGS.iters+FLAGS.warmup_iters):
    if i == FLAGS.warmup_iters:
      start_time = time.time()
    # fetch value into Python runtime
    if direction == "t->p":
      result = sess.run(params_read)
      if FLAGS.sanity_check:
        total += result.sum()
        print(float(total)/params_size)
    elif direction == "p->t":
      sess.run(params_write.op, feed_dict={params_holder: result})
      

  elapsed_time = time.time() - start_time
  rate = float(FLAGS.iters)*FLAGS.data_mb/elapsed_time
  print("%5s %.2f MB/second" % (direction, rate))
  sess.run(done_queue.enqueue(1))
  return rate

def create_done_queue(i):
  """Queue used to signal death for i'th worker."""
  
  return tf.FIFOQueue(1, tf.int32, shared_name="done_queue"+
                      str(i))

if __name__ == '__main__':
  # run local benchmark in launcher and launch service
  if FLAGS.worker_type == "launcher":
    run_benchmark("")  # run local benchmark in launcher
    if FLAGS.profile:
      gc.collect()
      launch_distributed_service()

  # run distributed benchmark in client
  elif FLAGS.worker_type == "client":
    if not FLAGS.in_process:
      run_benchmark("grpc://%s:%s"%(host, FLAGS.port))
    log("Killing worker.")

  elif FLAGS.worker_type == "worker": # run tensorflow worker
    server = tf.train.Server(clusterspec(), config=session_config(),
                             job_name="worker",
                             task_index=0)
    queue = create_done_queue(0)
    sess = tf.Session(server.target, config=session_config())
    if FLAGS.in_process:
      run_benchmark(server.target)

    sess.run(queue.dequeue())
    time.sleep(2) # give chance for master session.run call to return
    if FLAGS.verbose:
      print("Worker %s quitting." %(FLAGS.task_index))
  else:
    assert False, "Unknown worker type"
