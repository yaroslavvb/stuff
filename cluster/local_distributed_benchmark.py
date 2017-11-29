"""Benchmark tensorflow distributed by adding vector of ones on worker2
to variable on worker1 as fast as possible.
On 2014 macbook, TensorFlow 0.10 this shows
Local rate:       2175.28 MB per second
Distributed rate: 107.13 MB per second
"""

import subprocess
import tensorflow as tf
import time
import sys

flags = tf.flags
flags.DEFINE_integer("iters", 10, "Maximum number of additions")
flags.DEFINE_integer("data_mb", 100, "size of vector in MBs")
flags.DEFINE_string("port1", "12224", "port of worker1")
flags.DEFINE_string("port2", "12225", "port of worker2")
flags.DEFINE_string("task", "", "internal use")
FLAGS = flags.FLAGS

# setup local cluster from flags
host = "127.0.0.1:"
cluster = {"worker": [host+FLAGS.port1, host+FLAGS.port2]}
clusterspec = tf.train.ClusterSpec(cluster).as_cluster_def()

def default_config():
  optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)
  config = tf.ConfigProto(
    graph_options=tf.GraphOptions(optimizer_options=optimizer_options))
  config.log_device_placement = False
  config.allow_soft_placement = False
  return config

def create_graph(device1, device2):
  """Create graph that keeps variable on device1 and
  vector of ones/addition op on device2"""
  
  tf.reset_default_graph()
  dtype=tf.int32
  params_size = 250*1000*FLAGS.data_mb # 1MB is 250k integers

  with tf.device(device1):
    params = tf.get_variable("params", [params_size], dtype,
                             initializer=tf.zeros_initializer)
  with tf.device(device2):
    # constant node gets placed on device1 because of simple_placer
    #    update = tf.constant(1, shape=[params_size], dtype=dtype)
    update = tf.get_variable("update", [params_size], dtype,
                             initializer=tf.ones_initializer)
    add_op = params.assign_add(update)
    
  init_op = tf.initialize_all_variables()
  return init_op, add_op

def run_benchmark(sess, init_op, add_op):
  """Returns MB/s rate of addition."""
  
  sess.run(init_op)
  sess.run(add_op.op)  # warm-up
  start_time = time.time()
  for i in range(FLAGS.iters):
    # change to add_op.op to make faster
    sess.run(add_op)
  elapsed_time = time.time() - start_time
  return float(FLAGS.iters)*FLAGS.data_mb/elapsed_time


def run_benchmark_local():
  ops = create_graph(None, None)
  sess = tf.Session(config=default_config())
  return run_benchmark(sess, *ops)


def run_benchmark_distributed():
  ops = create_graph("/job:worker/task:0", "/job:worker/task:1")

  # launch distributed service
  def runcmd(cmd): subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT)
  runcmd("python %s --task=0"%(sys.argv[0]))
  runcmd("python %s --task=1"%(sys.argv[0]))
  time.sleep(1)

  sess = tf.Session("grpc://"+host+FLAGS.port1, config=default_config())
  return run_benchmark(sess, *ops)
  
if __name__=='__main__':
  if not FLAGS.task:

    rate1 = run_benchmark_local()
    rate2 = run_benchmark_distributed()

    print("Adding data in %d MB chunks" %(FLAGS.data_mb))
    print("Local rate:       %.2f MB per second" %(rate1,))
    print("Distributed rate: %.2f MB per second" %(rate2,))

  else: # Launch TensorFlow server
    server = tf.train.Server(clusterspec, config=default_config(),
                             job_name="worker",
                             task_index=int(FLAGS.task))
    server.join()
