#!/usr/bin/env python

# ImageNet experiments
# 1 worker, 1 ps: 1 gpu/machine
# python launch_async_adder.py --cluster=aws --run=gpu
# worker 0 log:
# 1	images/sec: 52.5 +/- 0.0 (jitter = 0.0)	8.128
# 10	images/sec: 52.3 +/- 0.0 (jitter = 0.1)	8.093
# 20	images/sec: 52.2 +/- 0.0 (jitter = 0.1)	8.132
# 30	images/sec: 52.2 +/- 0.0 (jitter = 0.2)	8.023

# 1 worker, 1 ps: 8 gpus/machine
# python launch_async_adder.py --cluster=aws --run=gpu2
# worker 0 log:
# 1	images/sec: 391.0 +/- 0.0 (jitter = 0.0)	8.070
# 10	images/sec: 391.7 +/- 0.5 (jitter = 2.1)	7.985
# 20	images/sec: 391.0 +/- 0.3 (jitter = 1.1)	7.969
# 30	images/sec: 390.5 +/- 0.3 (jitter = 1.0)	7.958
# 40	images/sec: 389.9 +/- 0.3 (jitter = 1.4)	7.990

# 2 workers, 1 ps
# python launch_async_adder.py --cluster=aws --run=gpu4 --num_workers=2 --num_ps=1
# worker 0 log:
# 1	images/sec: 385.2 +/- 0.0 (jitter = 0.0)	8.014
# 10	images/sec: 383.4 +/- 0.5 (jitter = 1.1)	7.928
# 20	images/sec: 383.3 +/- 0.4 (jitter = 2.1)	7.910
# 30	images/sec: 382.6 +/- 0.6 (jitter = 2.2)	7.884
# 40	images/sec: 381.9 +/- 0.5 (jitter = 2.4)	7.914

# 5 workers, 1 ps
# ./launch_async_adder.py --run=gpu5 --cluster=aws --num_workers=5 --num_ps=4 \
# --worker_type=p2.8xlarge --ps_type=c5.large
# worker 0 log:
# 1	images/sec: 384.1 +/- 0.0 (jitter = 0.0)	7.943
# 10	images/sec: 381.8 +/- 3.4 (jitter = 2.0)	7.868
# 20	images/sec: 382.6 +/- 1.8 (jitter = 3.3)	7.861
# 30	images/sec: 382.7 +/- 1.2 (jitter = 3.1)	7.854
#
#
# 8 workers, 4 ps
# ./launch_async_adder.py --run=cnn --num_workers=8 --num_ps=4 --worker_type=p2.8xlarge --ps_type=c5.2xlarge
#
# worker 0 log:
# 1	images/sec: 389.6 +/- 0.0 (jitter = 0.0)	7.903
# 10	images/sec: 388.4 +/- 0.7 (jitter = 1.2)	7.867
# 20	images/sec: 388.3 +/- 0.4 (jitter = 0.7)	7.851
# 30	images/sec: 387.9 +/- 0.3 (jitter = 1.6)	7.863
#
#
# after enabling placement groups
# python ./launch_async_adder.py --run=cnn2 --num_workers=8 --num_ps=4 --worker_type=p2.8xlarge --ps_type=c5.2xlarge
# worker 0 log:
# 1	images/sec: 368.4 +/- 0.0 (jitter = 0.0)	7.925
# 10	images/sec: 368.7 +/- 0.8 (jitter = 1.3)	7.880
# 20	images/sec: 368.5 +/- 0.5 (jitter = 0.8)	7.873
#
#
# after disabling placement groups again
# python ./launch_async_adder.py --run=cnn2 --num_workers=8 --num_ps=4 --worker_type=p2.8xlarge --ps_type=c5.2xlarge --disable_placement
# 1	images/sec: 371.8 +/- 0.0 (jitter = 0.0)	7.928
# 10	images/sec: 370.0 +/- 0.6 (jitter = 1.6)	7.886
# 20	images/sec: 369.1 +/- 0.4 (jitter = 2.1)	7.874
# 30	images/sec: 369.0 +/- 0.3 (jitter = 1.6)	7.886
# 40	images/sec: 368.4 +/- 0.3 (jitter = 1.6)	7.914


# Works either with remote or local instances.
# Local instances are tmux sessions. Each session has separate window
# corresponding to the task.

# Remote instances are on AWS. If given job name exists, it will assume
# it has correct number of instances (tasks), and reuse those instances.

# Job naming:
# test-ps (2 instances)
# test-worker (3 instances)
# test-tb (tensorboard process)

# Locally
# tmux session test-ps
# tmux session test-worker

# todo: utility to scp directory to amazon machine

# TODO: check if instance limits are satisfiable ahead of time, crash early if
# not (need iam:GetAccountSummary  permission)
#
# Stop failure that happens when cluster init is triggered before instances
# are ready, currently stack trace as below
#     self.stop(close_summary_writer=close_summary_writer)
#   File "/home/ubuntu/anaconda3/envs/py2/lib/python2.7/site-packages/tensorflow/python/training/supervisor.py", line 820, in stop
#     ignore_live_threads=ignore_live_threads)
#   File "/home/ubuntu/anaconda3/envs/py2/lib/python2.7/site-packages/tensorflow/python/training/coordinator.py", line 387, in join
#     six.reraise(*self._exc_info_to_raise)
#   File "/home/ubuntu/anaconda3/envs/py2/lib/python2.7/site-packages/tensorflow/python/training/supervisor.py", line 981, in managed_session
#     start_standard_services=start_standard_services)
#   File "/home/ubuntu/anaconda3/envs/py2/lib/python2.7/site-packages/tensorflow/python/training/supervisor.py", line 726, in prepare_or_wait_for_session
#     max_wait_secs=max_wait_secs)
#   File "/home/ubuntu/anaconda3/envs/py2/lib/python2.7/site-packages/tensorflow/python/training/session_manager.py", line 400, in wait_for_session
#     sess)
#   File "/home/ubuntu/anaconda3/envs/py2/lib/python2.7/site-packages/tensorflow/python/training/session_manager.py", line 481, in _try_run_local_init_op
#     is_ready_for_local_init, msg = self._model_ready_for_local_init(sess)
#   File "/home/ubuntu/anaconda3/envs/py2/lib/python2.7/site-packages/tensorflow/python/training/session_manager.py", line 466, in _model_ready_for_local_init
#     "Model not ready for local init")
#   File "/home/ubuntu/anaconda3/envs/py2/lib/python2.7/site-packages/tensorflow/python/training/session_manager.py", line 508, in _ready
#     ready_value = sess.run(op)
#   File "/home/ubuntu/anaconda3/envs/py2/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 889, in run
#     run_metadata_ptr)
#   File "/home/ubuntu/anaconda3/envs/py2/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 1120, in _run
#     feed_dict_tensor, options, run_metadata)
#   File "/home/ubuntu/anaconda3/envs/py2/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 1317, in _do_run
#     options, run_metadata)
#   File "/home/ubuntu/anaconda3/envs/py2/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 1336, in _do_call
#     raise type(e)(node_def, op, message)
# tensorflow.python.framework.errors_impl.UnavailableError: Endpoint read failed

import base64
import json
import os
import portpicker
import shlex
import subprocess
import sys
import tensorflow as tf
import threading
import time
from collections import defaultdict
import pickle

module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, module_path+'/tf-tools/benchmark/runner')
import cluster_aws

import aws
import tmux

# TODO: stop launcher script from quitting when ssh command returns
# (remove daemon thread designation)

# todo: robustness to worker restarts

# todo, way to do "tail -f" to show logs from single task


################################################################################
# User-specific AWS config
AMI = 'ami-9ddb0fe5'
KEY_NAME = 'yaroslav'  # AWS key-name to use
KEY_PATH = os.environ['HOME']+'/d/yaroslav.pem' # location of .pem file on
                                                # local filesystem
SECURITY_GROUP = 'open' # security group for all instances
################################################################################


LOCAL_LOGDIR_PREFIX='/temp/logs'
EFS_LOGDIR_PREFIX='/efs/logs'

# TODO: replace tmux_name with just name
flags = tf.flags
flags.DEFINE_string('run', 'default',
                    'tag used to keep track of machines in this experiment')
flags.DEFINE_integer("num_workers", 1, "number of gradient workers")
flags.DEFINE_integer("num_ps", 1, "number of ps workers")
#flags.DEFINE_boolean("verbose", False, "whether to have verbose logging")
flags.DEFINE_string("tmux_name", "async_adder", "name to use for tmux session")
#flags.DEFINE_string('localdir_prefix', '/temp/stdout',
#                     'where to mirror worker logs locally')
#flags.DEFINE_string('logdir_prefix', '/efs/logs',
#                     'where to dump EFS logs')
flags.DEFINE_integer('remote_worker_port', 3333, 'port to use for '
                     'remote connections')
flags.DEFINE_integer('remote_tb_port', 6006, 'port to use for '
                     'tensorboard service')
flags.DEFINE_string("cluster", "aws", "where to run (aws or local)")
flags.DEFINE_boolean("kill", False, "brings down experiment")
flags.DEFINE_string('worker_type', 'p2.8xlarge', 'instance type to use for workers')
flags.DEFINE_string('ps_type', 'c5.large', 'instance type to use for ps')
flags.DEFINE_boolean('disable_placement', False, 'disable placement groups')

FLAGS = flags.FLAGS


WORKER_CMD='python ./async_adder.py'  # todo: path robustness?
PS_CMD='python ./async_adder.py'


# TODO: worker/ps logs are different, make sure different worker tasks
# are actually writing to different logs

# todo: sometimes get 'endpoint read failed', due to AWS not bringing up
# all instances yet perhaps
def ossystem(cmd):
  print(cmd)
  os.system(cmd)

# todo: factor out into tmux_lib


def setup_local_logdir(run):
  logdir = LOCAL_LOGDIR_PREFIX + '/' + run
  os.system('rm -Rf '+logdir)
  os.system('mkdir -p '+logdir)
  return logdir

def setup_remote_logdir(run):
  logdir = EFS_LOGDIR_PREFIX + '/' + run
  # TODO: make these mkdir's work
  #os.system('rm -Rf '+logdir)
  #os.system('mkdir -p '+logdir)
  return logdir

initialized_windows = set()
def run_in_window(window, cmd_list):
  """Runs command in tmux window, initializing tmux session and window if
    necessary. cmd_list is list of args"""

  if isinstance(cmd_list, str):
    cmd_list = [cmd_list]

  assert isinstance(cmd_list, list)
  
  global initialized_windows
  def run(cmd):
    ossystem("tmux send-keys -t {} '{}' Enter".format(window, cmd))

  # if nothing initialized, restart session
  if not initialized_windows:
    ossystem('tmux kill-session -t ' + FLAGS.tmux_name)
    # -d starts new session in detached mode
    # since can't start windowless tmux, start with dummy window and rename
    ossystem('tmux new-session -s %s -n %s -d '% (FLAGS.tmux_name, "blargh"))
    
  if not window in initialized_windows:
    if not initialized_windows:
      ossystem('tmux rename-window -t blargh '+window)
    else:
      ossystem("tmux new-window -t {} -n {}".format(FLAGS.tmux_name, window))

    initialized_windows.add(window)
    
  for cmd in cmd_list:
    run(cmd)

# all instances->tasks
def launch_job_tmux(role, num_tasks):
  
  job_name = FLAGS.run + '-'+role
  DEFAULT_NAME = 'blargh'
  ossystem('tmux kill-session -t ' + job_name)

  # todo: move tmux initialization into localjob init
  # TODO: don't need default name
  ossystem('tmux new-session -s %s -n %s -d '% (job_name, DEFAULT_NAME))
  ossystem('tmux rename-window -t %s %s '%(DEFAULT_NAME, '0'))
  for task_id in range(1, num_tasks):
    ossystem("tmux new-window -t {} -n {}".format(job_name, task_id))

  job = LocalJob(job_name, num_tasks)
  # setup environment
  for task in job.tasks:
    task.run('source activate sep22')
    
  return job


class LocalJob:
  def __init__(self, name, num_tasks):
    self.name = name
    self.num_tasks = num_tasks
    self.tasks = []
    for task_id in range(num_tasks):
      self.tasks.append(LocalTask(self, task_id))


class LocalTask: # same as Instance
  """Local tasks interacts with tmux session where session name is derived
  from job name, and windows are task ids."""

  def __init__(self, job, task_id):
    self.job = job
    self.ip = '127.0.0.1' # hostname/ip address
    self.id = task_id
    self.port = portpicker.pick_unused_port()

  def run(self, cmd):
    window = self.job.name+":"+str(self.id)
    ossystem("tmux send-keys -t {} '{}' Enter".format(window, cmd))


  def tf_env_setup(self, full_cluster_spec, task_spec):
    # full cluster config
    # todo: not needed
    #    cluster_config = {'cluster': cluster_spec, 'task': task_spec}

    task_type = task_spec['type']
    task_id = task_spec['index']
    print("Task id is %r"%(task_id,))
    host = full_cluster_spec[task_type][task_id]

    # every worker needs its own location
    sparse_cluster_spec = defaultdict(dict)
    sparse_cluster_spec[task_type][task_id] = host
    
    # worker workers know about all ps workers
    if task_type == 'worker':
      sparse_cluster_spec['ps'] = full_cluster_spec['ps']
      
    # ps workers know about all worker workers
    if task_type == 'ps':
      pass
      sparse_cluster_spec['worker'] = full_cluster_spec['worker']
      #sparse_cluster_spec['worker'] = {0: full_cluster_spec['worker'][0]}

    sparse_cluster_config = {'cluster': sparse_cluster_spec,
                             'task': task_spec}
    print("Cluster config for %s %s is %s"%(task_type, task_id,
                                            sparse_cluster_spec))
    json_string = json.dumps(sparse_cluster_config)
    json_string_encoded = base64.b16encode(json_string.encode('ascii'))
    json_string_encoded = json_string_encoded.decode('ascii')
    export_command = "export TF_CONFIG_BASE16=%s"%(json_string_encoded,)
    self.run(export_command)

    # json has problem with sparse clusterspec (0 can't be key, only "0")
    # therefore also dump clusterspec as pickle object
    pickle_string = pickle.dumps(sparse_cluster_config)
    pickle_string_encoded = base64.b16encode(pickle_string)
    pickle_string_encoded = pickle_string_encoded.decode('ascii')
    export_command = "export TF_PICKLE_BASE16=%s"%(pickle_string_encoded,)
    self.run(export_command)
    
    logdir = LOCAL_LOGDIR_PREFIX + '/' + FLAGS.run
    self.run("export LOGDIR="+logdir)


def select_window(window):
  """select the window to be in the foreground"""
  ossystem('tmux select-window -t %s:%s'% (FLAGS.tmux_name, window))

def launch_job_aws(name, replicas):

  # todo: rename instance_tag to name
  instances = cluster_aws.CreateAwsInstances(num_instances=num_instances,
                                             image_id=AMI,
                                             key_name=KEY_NAME,
                                             ssh_key=KEY_PATH,
                                             security_group=SECURITY_GROUP,
                                             instance_tag=name,
                                             placement_group='',
                                             instance_type=INSTANCE_TYPE)
  job = AwsJob()



def tf_config_cmd(full_cluster_spec, task_spec):
  task_type = task_spec['type']
  task_id = task_spec['index']
  print("Task id is %r"%(task_id,))
  host = full_cluster_spec[task_type][task_id]

  # every worker needs its own location
  sparse_cluster_spec = defaultdict(dict)
  sparse_cluster_spec[task_type][task_id] = host

  # worker workers know about all ps workers
  if task_type == 'worker':
    sparse_cluster_spec['ps'] = full_cluster_spec['ps']

  # ps workers know about all worker workers
  if task_type == 'ps':
    pass
    sparse_cluster_spec['worker'] = full_cluster_spec['worker']
    #sparse_cluster_spec['worker'] = {0: full_cluster_spec['worker'][0]}

  sparse_cluster_config = {'cluster': sparse_cluster_spec,
                           'task': task_spec}
  print("Cluster config for %s %s is %s"%(task_type, task_id,
                                          sparse_cluster_spec))
  json_string = json.dumps(sparse_cluster_config)
  json_string_encoded = base64.b16encode(json_string.encode('ascii'))
  json_string_encoded = json_string_encoded.decode('ascii')

  pickle_string = pickle.dumps(sparse_cluster_config)
  pickle_string_encoded = base64.b16encode(pickle_string)
  pickle_string_encoded = pickle_string_encoded.decode('ascii')
  export_command = "export TF_PICKLE_BASE16=%s"%(pickle_string_encoded,)
  return export_command

class BasicAwsJob():
  def __init__(name, num_tasks):
    pass

  
  
def launch_aws():
  ps_job = aws.tf_job(FLAGS.run+'-ps', FLAGS.num_ps)
  worker_job = aws.tf_job(FLAGS.run+'-worker', FLAGS.num_workers, placement_group='tf')
  tb_job = aws.tf_job(FLAGS.run+'-tb', 1, placement_group='tf')

  # wait for everything to come up

  # todo: private IP's may be known before instances are ready
  ps_job.wait_for_ready()
  worker_job.wait_for_ready()
  
  # TODO: orchestration may be easier if I save server spec to a predictable
  # location on AWS rather than passing it to each worker through command-line
  
  # Orchestration: every worker needs to know:
  # 1. their own role (task_spec), ie {type: worker, index: 0}
  # 2. role->ip mapping of all machines (cluster_spec), ie
  #    {"worker": ["localhost:24724"], "ps": ["localhost:15960"]}}
  ps_hosts = ["%s:%d"%(task.ip, task.port) for task in ps_job.tasks]
  worker_hosts = ["%s:%d"%(task.ip, task.port) for task in worker_job.tasks]
  cluster_spec = {'worker': worker_hosts, 'ps': ps_hosts}

  # launch parameter server tasks
  task_type = 'ps'  
  for task in ps_job.tasks:
    task_spec = {'type': task_type, 'index': task.id}
    task.run(generate_tf_env_setup_cmd(cluster_spec, task_spec))
    task.run(PS_CMD)

  # launch worker tasks
  task_type = 'worker' # task type can also be "chief", overlapping with worker
  for task in worker_job.tasks:
    task_spec = {'type': task_type, 'index': task.id}
    task.run(generate_tf_env_setup_cmd(cluster_spec, task_spec))
    task.run(WORKER_CMD)

  # launch tensorboard visualizer
  task = tb_job.tasks[0]
  task.run('tensorboard --port=%d --logdir=%s'%(task.port, logdir))

  

class Instance:
  # todo: move inside instance
  
  def tf_env_setup(self, cluster_spec, task_spec):
    cluster_config = {'cluster': cluster_spec, 'task': task_spec}
    json_string = json.dumps(cluster_config)
    json_string_encoded = base64.b16encode(json_string.encode('ascii'))
    json_string_encoded = json_string_encoded.decode('ascii')
    export_command = "export TF_CONFIG_BASE16=%s"%(json_string_encoded,)
    self.run(export_command)

# TODO: rename .ip to get_ip()

def launch_local():
  ps_job = launch_job_tmux('ps', FLAGS.num_ps)
  worker_job = launch_job_tmux('worker', FLAGS.num_workers)
  tb_job = launch_job_tmux('tb', 1)

  logdir = setup_local_logdir(FLAGS.run)

  # Orchestration: every worker needs to know:
  # 1. their own role (task_spec), ie {type: worker, index: 0}
  # 2. role->ip mapping of all machines (cluster_spec), ie
  #    {"worker": ["localhost:24724"], "ps": ["localhost:15960"]}}
   
  ps_hosts = ["%s:%d"%(task.ip, task.port) for task in ps_job.tasks]
  worker_hosts = ["%s:%d"%(task.ip, task.port) for task in worker_job.tasks]
  cluster_spec = {'worker': worker_hosts, 'ps': ps_hosts}

  # launch parameter server tasks
  task_type = 'ps'  
  for task in ps_job.tasks:
    task_spec = {'type': task_type, 'index': task.id}
    task.tf_env_setup(cluster_spec, task_spec)
    task.run(PS_CMD)

  # launch worker tasks
  task_type = 'worker' # task type can also be "chief", overlapping with worker
  for task in worker_job.tasks:
    task_spec = {'type': task_type, 'index': task.id}
    task.tf_env_setup(cluster_spec, task_spec)
    task.run(WORKER_CMD)

  # launch tensorboard visualizer
  task = tb_job.tasks[0]
  task.run('tensorboard --port=%d --logdir=%s'%(task.port, logdir))


# def launch_remote():
#   ps_job = AwsJob('ps', 1)
#   worker_job = launch_instances_aws('worker', 1)
#   tb_job = launch_instances_aws('tb', 1)[0]

#   # Orchestration, every worker needs to know their own role (task_spec)
#   # and role->ip mapping of all machines (cluster_spec)
#   # This information is saved as base16 encoded dictionary in env var
#   # and loaded in the task script
#   PORT = 3000
#   ps_hosts = ["%s:%d"%(i.ip, PORT) for i in ps_instances]
#   worker_hosts = ["%s:%d"%(i.ip, PORT) for i in worker_instances]
#   cluster_spec = {'worker': worker_hosts, 'ps': ps_hosts}

#   def tf_config_setup():
#     task_spec = {'type': task_type, 'index': task_id}
#     cluster_config = {'cluster': cluster_spec, 'task': task_spec}
#     json_string = json.dumps(cluster_config)
#     json_string_encoded = base64.b16encode(json_string.encode('ascii'))
#     json_string_encoded = json_string_encoded.decode('ascii')
#     export_command = "export TF_CONFIG_BASE16=%s"%(json_string_encoded,)
#     print('export command')
#     print(repr(export_command))
#     return export_command

#   task_type = 'ps'
#   for task_id, instance in enumerate(ps_instances):
#     instance.run(tf_config_setup())
#     instance.run(PS_CMD)

#   task_type = 'worker'
#   for task_id, instance in enumerate(worker_instances):
#     instance.run(tf_config_setup())
#     instance.run(WORKER_CMD)

#   tb_instance.run('tensorboard --port=%d --logdir=%s'%(port, logdir))

def cnn_launcher():
  """Experiment launcher."""
  if FLAGS.cluster == 'local':
    if FLAGS.kill:
      tmux.kill_job("ps")
      tmux.kill_job("worker")
      tmux.kill_job("tb")
      return
    
    logdir = setup_local_logdir(FLAGS.run)
    ps_job = tmux.tf_job('ps', FLAGS.num_ps)
    worker_job = tmux.tf_job('worker', FLAGS.num_workers)
    tb_job = tmux.tf_job('tb', 1)


    # Orchestration: every worker needs to know:
    # 1. their own role (task_spec), ie {type: worker, index: 0}
    # 2. role->ip mapping of all machines (cluster_spec), ie
    #    {"worker": ["localhost:24724"], "ps": ["localhost:15960"]}}

    ps_hosts = ["%s:%d"%(task.ip, task.port) for task in ps_job.tasks]
    worker_hosts = ["%s:%d"%(task.ip, task.port) for task in worker_job.tasks]
    cluster_spec = {'worker': worker_hosts, 'ps': ps_hosts}

    # launch parameter server tasks
    task_type = 'ps'  
    for task in ps_job.tasks:
      task_spec = {'type': task_type, 'index': task.id}
      task.run(tf_config_cmd(cluster_spec, task_spec))
      task.run("export LOGDIR="+logdir)
      task.run(PS_CMD)

    # launch worker tasks
    task_type = 'worker'
    for task in worker_job.tasks:
      task_spec = {'type': task_type, 'index': task.id}
      task.run(tf_config_cmd(cluster_spec, task_spec))
      task.run("export LOGDIR="+logdir)
      task.run(WORKER_CMD)

    # launch tensorboard visualizer
    task = tb_job.tasks[0]
    task.run("export LOGDIR="+logdir)
    task.run('tensorboard --port=%d --logdir=$LOGDIR')
  elif FLAGS.cluster == 'aws':

    # create placement group, same as run name
    import boto3
    ec2 = boto3.client('ec2')
    
    if not FLAGS.disable_placement:
      placement_group = FLAGS.run
      try:
        response = ec2.create_placement_group(GroupName=placement_group,
                                            Strategy='cluster')
      except Exception as e:
        if 'Duplicate' in e.response['Error']['Code']:
          print("Warning, placement group %s already exists, skipping" %(placement_group,))
          print("Got message "+str(e))
    else:
      placement_group = ''


    # out of instances for c5.large
    logdir = setup_remote_logdir(FLAGS.run)
    ps_job = aws.tf_job(FLAGS.run+'/ps', FLAGS.num_ps,
                        instance_type=FLAGS.ps_type,
                        placement_group=placement_group)
    worker_job = aws.tf_job(FLAGS.run+'/worker', FLAGS.num_workers,
                            instance_type=FLAGS.worker_type,
                            placement_group=placement_group)

    ps_job.wait_until_ready()
    worker_job.wait_until_ready()
    #    tb_job = aws.tf_job('tb', 1, instance_type='m2.micro')

    # Orchestration: every worker needs to know:
    # 1. their own role (task_spec), ie {type: worker, index: 0}
    # 2. role->ip mapping of all machines (cluster_spec), ie
    #    {"worker": ["localhost:24724"], "ps": ["localhost:15960"]}}

    ps_hosts = ["%s:%d"%(task.ip, task.port) for task in ps_job.tasks]
    ps_hosts_str = ','.join(ps_hosts)
    worker_hosts = ["%s:%d"%(task.ip, task.port) for task in worker_job.tasks]
    worker_hosts_str = ','.join(worker_hosts)
    cluster_spec = {'worker': worker_hosts, 'ps': ps_hosts}

    setup_cmd = "source ~/.bashrc && export PATH=~/anaconda3/bin:$PATH && source activate py2 && cd ~/git0/benchmarks/scripts/tf_cnn_benchmarks"


    for task in ps_job.tasks:
      task.run("killall python")
    for task in worker_job.tasks:
      task.run("killall python")

    time.sleep(5)
    
    # launch parameter server tasks
    task_type = 'ps'
    cmds = []
    ps_cmd_tmpl = "CUDA_VISIBLE_DEVICES='' python tf_cnn_benchmarks.py --local_parameter_device=gpu --worker_hosts=%(worker_hosts)s --ps_hosts=%(ps_hosts)s --job_name=ps --task_index=%(task_index)s"
    for task in ps_job.tasks:
      cmds = []
      task_spec = {'type': task_type, 'index': task.id}
      cmds.append(setup_cmd)
      cmds.append(tf_config_cmd(cluster_spec, task_spec))
      cmds.append("export LOGDIR="+logdir)
      #      task.upload("./benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py")
      task.upload("./benchmarks/scripts/tf_cnn_benchmarks/variable_mgr.py",
                  "/home/ubuntu/Dropbox/git0/benchmarks/scripts/tf_cnn_benchmarks/variable_mgr.py")
      cmds.append(ps_cmd_tmpl % {"worker_hosts": worker_hosts_str,
                                 "ps_hosts": ps_hosts_str,
                                 "job_name": task_type,
                                 "task_index": task.id})
      task.run(' && '.join(cmds))
      print("To see the output: tail -f %s" %(task.last_stdout))

    # launch worker tasks
    task_type = 'worker'
    cmds = []
    worker_cmd_tmpl = "python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=64 --num_batches=1000 --model=resnet50 --optimizer=sgd --variable_update=distributed_replicated --cross_replica_sync=True --local_parameter_device=gpu --num_gpus=8 --nodistortions --display_every=10 --worker_hosts=%(worker_hosts)s --ps_hosts=%(ps_hosts)s --job_name=worker --task_index=%(task_index)s"

    for task in worker_job.tasks:
      cmds = []
      task_spec = {'type': task_type, 'index': task.id}
      cmds.append(setup_cmd)

      cmds.append(tf_config_cmd(cluster_spec, task_spec))
      cmds.append("export LOGDIR="+logdir)
      task.upload("./benchmarks/scripts/tf_cnn_benchmarks/variable_mgr.py",
                  "/home/ubuntu/Dropbox/git0/benchmarks/scripts/tf_cnn_benchmarks/variable_mgr.py")
      #      task.upload("./benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py")
      cmds.append(worker_cmd_tmpl % {"worker_hosts": worker_hosts_str,
                                     "ps_hosts": ps_hosts_str,
                                     "job_name": task_type,
                                     "task_index": task.id})
      task.run(' && '.join(cmds))
      print("To see the output of %s: tail -f %s" %(task.id,
                                                    task.last_stdout))


    print("Sleeping")
    time.sleep(1000)

    # TODO: are ssh calls blocking?
    
    # launch tensorboard visualizer
    #    task = tb_job.tasks[0]
    #    cmds = []
    #    cmds.append("export LOGDIR="+logdir)
    #    cmds.append('tensorboard --port=%d --logdir=$LOGDIR' %(task.port))
    #    task.run(' && '.join(cmds))
    
    

def main():
  os.system('rm -Rf data') # todo: remove
  cnn_launcher()
  

if __name__=='__main__':
  main()
  
