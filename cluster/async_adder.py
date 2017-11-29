#!/usr/bin/env python
import base64
import os
import portpicker
import subprocess
import sys
import tensorflow as tf
import threading
import time
import pickle

from tensorflow.python.summary import summary as summary_lib
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.util import compat
from tensorflow.core.util import event_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.python.training import training_util # TOOD: not needed?

from tensorflow.python.framework import device as pydev

from myutil import timeit

# TODO: when ps server restarts, it doesn't reinitialize the variables
# TODO: document TF_CONFIG

RETRY_DELAY_SEC = 5

# TODO: replace with "sharded"
flags = tf.flags
flags.DEFINE_integer("iters", 1000, "number of times to repeat experiment")
flags.DEFINE_integer("iters_per_step", 10, "number of additions per step")
flags.DEFINE_integer("data_mb", 128, "size of vector in MBs")
flags.DEFINE_boolean("verbose", False, "whether to have verbose logging")
flags.DEFINE_boolean("profile", False, "whether to collect CPU profile")

# internal flags, set by client
FLAGS = flags.FLAGS


# TODO: remove logdir prefix, it should be global settings that doesn't change
# todo: name not needed?
#flags.DEFINE_string('logdir', '', 'where to event logs')
flags.DEFINE_string('name', 'default',
                    'tag used to keep track of machines in this experiment')


# TODO: switch back to regular (not traced) runs
timeline_counter = 0
run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
def traced_run(fetches):
  """Runs fetches, dumps timeline files in current directory."""

  global timeline_counter
  run_metadata = tf.RunMetadata()

  config = load_config()
  log_fn = "%s-%s-%s"%(config.task_type, config.task_id, timeline_counter)
  sess = tf.get_default_session()
  
  root = os.getcwd()+"/data"
  os.system('mkdir -p '+root)
  
  from tensorflow.python.client import timeline

  results = sess.run(fetches,
                     options=run_options,
                     run_metadata=run_metadata);
  tl = timeline.Timeline(step_stats=run_metadata.step_stats)
  ctf = tl.generate_chrome_trace_format(show_memory=True,
                                          show_dataflow=False)
  open(root+"/timeline_%s.json"%(log_fn,), "w").write(ctf)
  open(root+"/stepstats_%s.pbtxt"%(log_fn,), "w").write(str(
    run_metadata.step_stats))
  timeline_counter+=1
  return results


def sessrun(fetches):
  sess = tf.get_default_session()
  return sess.run(fetches)
  return traced_run(fetches)


def get_ps_device(task=0, op_device_str=''):
  device_str = '/job:ps'
  device = pydev.DeviceSpec.from_string(device_str)
  device.task = task
  op_device = pydev.DeviceSpec.from_string(op_device_str)
  device.merge_from(op_device)
  return device.to_string()

# todo: private methods
def get_worker_device(task, op_device_str=''):
  device_str = '/job:worker'
  device = pydev.DeviceSpec.from_string(device_str)
  device.task = task
  op_device = pydev.DeviceSpec.from_string(op_device_str)
  device.merge_from(op_device)
  return device.to_string()

def session_config():
  optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)
  config = tf.ConfigProto(
    graph_options=tf.GraphOptions(optimizer_options=optimizer_options))
  
  config.operation_timeout_in_ms = 10*1000  # abort after 10 seconds
  return config

def write_event(tag, value, step):
  event = event_pb2.Event(
      wall_time=time.time(),
      step=step,
      summary=summary_pb2.Summary(
          value=[summary_pb2.Summary.Value(
              tag=tag, simple_value=value)]))

  # todo: not flush so often?
  writer.WriteEvent(event)
  writer.Flush()

  return event

def make_params():
  params_size = 250*1000*FLAGS.data_mb # 1MB is 250k integers
  dtype=tf.int32
  ps_device = get_ps_device(0)
  with tf.device(ps_device):
    params = tf.get_variable("params", [params_size], dtype,
                             initializer=tf.ones_initializer())
  return params
    
def run_worker():
  """Main worker loop."""

  # todo: rename "config" into distributed_config
  config = load_config()
  cluster_spec = config.cluster_spec
  #  import pdb; pdb.set_trace()

  ps_tasks = len(cluster_spec['ps'])
  assert ps_tasks >= 0

  # returns device like /job:worker/task:0
  worker_device = ''
  assert config.task_type == 'worker'
  
  if config.task_id == 1:
    time.sleep(60)  # slow-down second worker
  
  worker_device = get_worker_device(config.task_id)

  ps_device = get_ps_device(0)

  # todo: replace with int64
  # todo: replace with varscope.getvariable like in alextp suggestion
  with timeit("worker graph create"):
    params = make_params()
    with tf.device(worker_device):
      val = tf.ones((), dtype=params.dtype)
      grads = tf.fill([params.shape[0]], val)
      # todo: add two-way communication

    with tf.device(ps_device):
      update = params.assign_add(grads)
      params0 = params[0]

    #uninitialized_op = tf.report_uninitialized_variables()
    initialized_op = tf.is_variable_initialized(params)
  
  # todo: check how estimator does it
  # TODO: retries for errors during server creation?
  # it can fail if assigned port is unavailable
  with timeit("worker server start"):
    server = tf.train.Server(cluster_spec, config=session_config(),
                             job_name=config.task_type,
                             task_index=config.task_id)

    # follow logic in prepare_session
    # https://github.com/tensorflow/tensorflow/blob/22586bdf900640217deac6dc826054bc6e785518/tensorflow/python/training/session_manager.py#L71

  def create_session():
    #    uninited_list = ['somevariable']
    is_initialized = False
    while not is_initialized:
      try:
        with timeit("session creation"):
          sess = tf.InteractiveSession(server.target, config=session_config())
        with timeit("sessrun"):
          #          uninited_list = sessrun(uninitialized_op)
          is_initialized = sessrun(initialized_op)
      except Exception as e:
        print("Initialization failed with %s, retrying" %(e,))
      print(("Model not initialized, "
             "retrying in %.1f seconds" %(RETRY_DELAY_SEC,)))
      time.sleep(RETRY_DELAY_SEC)
    return sess
    
  # are there failures in creating session
  with timeit('create session'):
    sess = tf.InteractiveSession(server.target, config=session_config())
    
  # only run initialization on worker task 0
  if config.task_id == 0:
    sess_run_succeeded = False
    while not sess_run_succeeded:
      try:
        with timeit('intialize vars'):
          sessrun(params.initializer)
          sess_run_succeeded = True
      except Exception as e:
        print("Initialization failed with %s, retrying "
              "in %.1f sec" %(e, RETRY_DELAY_SEC))
        # this can fail if workers too too long to come up and
        # sessrun failed with DeadlineExceeded
        time.sleep(RETRY_DELAY_SEC)
    

  for step in range(FLAGS.iters):
    start_time = time.time()
    for i in range(FLAGS.iters_per_step):
      sess_run_succeeded = False
      while not sess_run_succeeded:
        try:
          sessrun(update)
          sess_run_succeeded = True
        # Exception when ps restarts, need to recreate session
        except Exception as e:  
          print(("sess run failed with %s, "
                 "retrying in %.1f seconds" %(e, RETRY_DELAY_SEC,)))
          time.sleep(RETRY_DELAY_SEC)
          sess = create_session()

    elapsed_time = time.time() - start_time
    rate = float(FLAGS.iters_per_step)*FLAGS.data_mb/elapsed_time
    event = write_event('rate', rate, step)
    print('%.2f MB/s'%(rate,))


# replacement of estimators.run_config.ClusterConfig that works with sparse
# cluster config

class MyClusterConfig:
  def __init__(self):
    self.task_id = -1
    self.task_type = "asdf"
    self.cluster_spec = {"asdf":"asdf"}

  def __str__(self):
    return self.__dict__.__str__()

def load_config():
  """Returns ClusterConfig object. Config contains task spec and cluster spec in dictionary-like form as below
  # {"task": {"index": 0, "type": "worker"}, "cluster": {"worker": ["localhost:24724"], "ps": ["localhost:15960"]}}
  """
  # old way that doesn't work for sparse format
  # if 'TF_CONFIG' not in os.environ:
  #   # try loading encoded version
  #   if 'TF_CONFIG_BASE16' in os.environ:
  #     tf_config_str = base64.b16decode(os.environ['TF_CONFIG_BASE16'])
  #     tf_config_str = tf_config_str.decode('ascii')
  #     os.environ['TF_CONFIG'] = tf_config_str
  #     del os.environ['TF_CONFIG_BASE16']
  #   else:
  #     assert False, "Must specify TF_CONFIG or TF_CONFIG_BASE16"
      
#  from tensorflow.contrib.learn.python.learn.estimators.run_config import ClusterConfig
  
  config = MyClusterConfig()
  config_dict = pickle.loads(base64.b16decode(os.environ["TF_PICKLE_BASE16"]))
  config.task_type = config_dict["task"]["type"]
  config.task_id = config_dict["task"]["index"]
  config.cluster_spec = config_dict["cluster"]
  return config

def run_ps():
  config = load_config()
  
  assert config.task_type == 'ps'
  params = make_params()
  
  with timeit('create server'):
    print("Starting server with target %s"%(config.cluster_spec[config.task_type][config.task_id]))
    server = tf.train.Server(config.cluster_spec, config=session_config(),
                             job_name=config.task_type,
                             task_index=config.task_id)

  # doing init run from ps master fails with
  # sess run failed with No worker known as /job:worker/replica:0/task:1
  #      [[Node: Fill_S3 = _Recv[client_terminated=false, recv_device="/job:ps/replica:0/task:0/device:CPU:0", send_device="/job:worker/replica:0/task:1/device:CPU:0", send_device_incarnation=7403937842608207616, tensor_name="edge_3_Fill", tensor_type=DT_INT32, _device="/job:ps/replica:0/task:0/device:CPU:0"]()]], retrying in 5.0 seconds

  # todo: replace with dequeue for graceful shutdown
  # todo: done_queue from sharded_ps_benchmark
  # done_queue = create_done_queue(0)
  time.sleep(365*24*3600)

def _get_master():
  """Returns the appropriate string for local grpc TensorFlow master.
  For compat with server.target, return bytes instead of string.

  The address is derived from server spec, so it may not match the value
  returned by server.target stared locally (server.target can be localhost:129)
  """

  def _get_master_str():
    config = load_config()
    task_type = config.task_type
    task_id = config.task_id
    cluster_spec = config.cluster_spec

    if not cluster_spec:
      return ''

    # If there is only one node in the cluster, do things locally.
    jobs = cluster_spec.jobs
    if len(jobs) == 1 and len(cluster_spec.job_tasks(jobs[0])) == 1:
      return ''

    # Lookup the master in cluster_spec using task_type and task_id,
    # if possible.
    if task_type:
      if task_type not in jobs:
        raise ValueError(
            '%s is not a valid task_type in the cluster_spec:\n'
            '%s\n\n'
            'Note that these values may be coming from the TF_CONFIG environment '
            'variable.' % (task_type, cluster_spec))
      addresses = cluster_spec.job_tasks(task_type)
      if task_id >= len(addresses) or task_id < 0:
        raise ValueError(
            '%d is not a valid task_id for task_type %s in the '
            'cluster_spec:\n'
            '%s\n\n'
            'Note that these value may be coming from the TF_CONFIG environment '
            'variable.' % (task_id, task_type, cluster_spec))
      return 'grpc://' + addresses[task_id]

    # For backwards compatibility, we return empty string if task_type was
    # not set (task_type did not previously exist).
    return ''

  return _get_master_str().encode('ascii')

def main():
  global writer
  config = load_config()

  # todo: factor out common logic
  logdir = os.environ["LOGDIR"]
  writer = pywrap_tensorflow.EventsWriter(compat.as_bytes(logdir+'/events'))

  if  config.task_type == 'worker':
    run_worker()
  elif config.task_type == 'ps':
    run_ps()
  else:
    assert False, "Unknown task type "+str(config.task_type)
    
  writer.Close()


if __name__=='__main__':
  main()
