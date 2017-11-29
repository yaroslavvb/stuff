
#import util as myutil
from collections import OrderedDict
from collections import defaultdict
from pprint import pprint as pp
import argparse
import base64
import base64
import boto3
import json
import os
import os
import pickle
import portpicker
import shlex
import struct
import subprocess
import sys
import sys
import tensorflow as tf
import threading
import threading
import time
import time
import yaml

LOCAL_TASKLOGDIR_PREFIX='/temp/tasklogs'
LOCAL_LOGDIR_PREFIX='/temp/logs'

def _setup_logdir(job_name):
  """Creates appropriate logdir for given job."""
  run_name = job_name.rsplit('-',1)[0]  # somerun-11-ps -> somerun-11
  logdir = '/temp/logs/'+run_name
  os.system('rm -Rf '+logdir)
  os.system('mkdir -p '+logdir)
  return logdir



def _ossystem(cmd):
  print(cmd)
  os.system(cmd)

def kill_job(name):
  """Simple local TensorFlow job launcher."""
  
  _ossystem('tmux kill-session -t ' + name)

def tf_job(name, num_tasks):
  """Simple local TensorFlow job launcher."""
  
  DEFAULT_NAME = '0'
  _ossystem('tmux kill-session -t ' + name)

  # TODO: don't need default name
  tmux_windows = [name+":"+str(0)]
  _ossystem('tmux new-session -s %s -n %s -d '% (name, DEFAULT_NAME))
  _ossystem('tmux rename-window -t %s %s '%(DEFAULT_NAME, '0'))
  for task_id in range(1, num_tasks):
    _ossystem("tmux new-window -t {} -n {}".format(name, task_id))
    tmux_windows.append(name+":"+str(task_id))

  # todo: remove num_tasks
  job = Job(name, num_tasks, tmux_windows)
  # setup environment
  for task in job.tasks:
    task.run('source activate sep22')

  # todo: logdir is shared across jobs, so set it up in experiment launcher
  _setup_logdir(name)
  job.logdir = _setup_logdir(name)

  return job


class Job:
  def __init__(self, name, num_tasks, tmux_windows):
    self.name = name
    #    self.num_tasks = num_tasks
    self.tasks = []
    for task_id, tmux_window in enumerate(tmux_windows):
      self.tasks.append(Task(tmux_window, self, task_id))


class Task:
  """Local tasks interacts with tmux session where session name is derived
  from job name, and windows are task ids."""

  def __init__(self, tmux_window, job, task_id):
    self.tmux_window = tmux_window
    self.job = job
    self.ip = '127.0.0.1' # hostname/ip address
    self.id = task_id
    self.port = portpicker.pick_unused_port()
    self.connect_instructions = 'tmux a -t '+self.tmux_window

    self.last_stdout = '<unavailable>'  # compatiblity with aws.py:Task
    self.last_stderr = '<unavailable>'

  def run(self, cmd):
    _ossystem("tmux send-keys -t {} '{}' Enter".format(self.tmux_window, cmd))

  def upload(self, cmd):  # compatiblity with aws.py:Task
    pass

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
