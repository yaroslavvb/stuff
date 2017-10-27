# Benchmark transferring data from TF into Python runtime
#
## Dependencies:
# portpicker (pip install portpicker)
# tcmalloc4 (sudo apt-get install google-perftools)
# TF 0.12  (for var.read_value(), ones_initializer())
#
# On Linux default malloc is slow
# sudo apt-get install google-perftools
# export LD_PRELOAD="/usr/lib/libtcmalloc.so.4"
#
# local session benchmark:
# 2014 MacBook (100 iters):
# 128MB --  3.56 GB/s
# 1024MB -- 1.96 GB/s
# 
# Xeon E5-2630 v3 @ 2.40GHz (100 iters):
# 128 MB -- 0.43 GB/s (default malloc)
# 128 MB -- 4-6.2 GB/s (tcmalloc)
# 1024 MB -- 4-5.97 GB/s (tcmalloc)
#
#
# distributed session + profiling on Xeon:
# python client_transfer_benchmark.py --profile
# 116.70 MB per second
# 151 MB per second (using AsProtoTensorContent patch)
# google-pprof `which python` /tmp/profile.out.client --svg > /tmp/profile.client
# google-pprof `which python` /tmp/profile.out.worker --svg > /tmp/profile.worker
#
#
# Profiling feeding: on Xeon:
# python client_transfer_benchmark.py --direction=p2t --profile
# 1577.86 MB per second
#  143.93 MB per second


import gc
import os
import portpicker
import subprocess
import sys
import tensorflow as tf
import threading
import time

flags = tf.flags
flags.DEFINE_integer("iters", 10, "Maximum number of additions")
flags.DEFINE_integer("warmup_iters", 1, "warmup iterations")
flags.DEFINE_integer("data_mb", 128, "size of vector in MBs")
flags.DEFINE_boolean("verbose", False, "extra logging")
flags.DEFINE_boolean("sanity_check", False, "run sanity check on results")
flags.DEFINE_boolean("profile", False, "whether to run distributed version and "
                                           "collect CPU profile")
flags.DEFINE_boolean("in_process", True, "do bencharmk on in-process master")
flags.DEFINE_string("direction", "t2p", "which direction to profile, either "
                    "tensorflow to python (t2p) or python to tensorflow (p2t)")

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

def run_benchmark(master):
  """Connect to master and run simple TF->Python transfer benchmark."""
  
  assert FLAGS.warmup_iters > 0
  gc.disable()
  
  dtype = tf.int32
  params_size = 250*1000*FLAGS.data_mb # 1MB is 250k integers
  params = tf.get_variable("params", [params_size], dtype,
                           initializer=tf.ones_initializer())
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
    if FLAGS.direction == "t2p":
      result = sess.run(params_read)
      if FLAGS.sanity_check:
        total += result.sum()
        print(float(total)/params_size)
    elif FLAGS.direction == "p2t":
      sess.run(params_write.op, feed_dict={params_holder: result})
      

  elapsed_time = time.time() - start_time
  rate = float(FLAGS.iters)*FLAGS.data_mb/elapsed_time
  print("%.2f MB per second" % (rate))
  sess.run(done_queue.enqueue(1))

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
