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


# If given job name exists, it will assume
# it has correct number of instances (tasks), and reuse those instances.

# without placement group
# python launch.py --run=cnn8 --num_workers=8 --num_ps=4 --worker_type=p2.8xlarge --ps_type=c5.2xlarge --disable_placement
# 1	images/sec: 373.4 +/- 0.0 (jitter = 0.0)	7.924
# 10	images/sec: 369.9 +/- 4.2 (jitter = 1.3)	7.888
# 20	images/sec: 369.6 +/- 3.0 (jitter = 1.5)	7.871
# 30	images/sec: 371.0 +/- 2.0 (jitter = 1.9)	7.884
# 40	images/sec: 371.5 +/- 1.5 (jitter = 1.7)	7.902
# 50	images/sec: 370.6 +/- 1.5 (jitter = 1.9)	7.866

# TODO: this can fail with
# tensorflow.python.framework.errors_impl.UnavailableError: Endpoint read failed

import base64
import json
import os
import pickle
import sys
import tensorflow as tf
import time

from collections import defaultdict

import aws

LOCAL_LOGDIR_PREFIX='/temp/logs'

flags = tf.flags
flags.DEFINE_string('run', 'default',
                    'tag used to keep track of machines in this experiment')
flags.DEFINE_integer("num_workers", 1, "number of gradient workers")
flags.DEFINE_integer("num_ps", 1, "number of ps workers")

# todo: remove
flags.DEFINE_string('worker_type', 'p2.8xlarge',
                    'instance type to use for gradient workers')
flags.DEFINE_string('ps_type', 'c5.large', 'instance type to use for '
                    'ps workers')

# started getting following error on placement groups, dsiable for now
# The placement group 'xyz' is in use in another availability zone: null
flags.DEFINE_boolean('disable_placement', True, 'disable placement groups')

FLAGS = flags.FLAGS

def ossystem(cmd):
  print(cmd)
  os.system(cmd)

# todo, move into aws.py
class AWSInstance(object):

  def __init__(self, instance, ssh_key='', name='', username='ubuntu',
               tags=None):
    # assert instance is aws instance
    self.aws_instance = instance
    self.ssh_key = ssh_key
    self.username = username
    if name:
      self.SetNameTag(name)
    if tags:
      for key, value in tags.items():
        self.SetCustomTag(key, value)

    self.opened_ssh_client = []

  def __del__(self):
    self.CleanSshClient()

  def WaitUntilReady(self):
    self.aws_instance.wait_until_running()
    # Sometimes the wait doesn't work, wait longer and check status
    client = boto3.client('ec2')
    while True:
      res = client.describe_instance_status(
          InstanceIds=[self.aws_instance.instance_id])
      try:
        if (res['InstanceStatuses'][0]['InstanceStatus']['Status'] == 'ok' and
            res['InstanceStatuses'][0]['SystemStatus']['Status'] == 'ok'):
          break
      except:
        print('instance has no status')
      time.sleep(30)

    self.aws_instance.load()
    self.hostname = self.aws_instance.public_dns_name

  def CreateSshClient(self):
    assert self.hostname is not None
    ssh_client = util.SshToHost(self.hostname, ssh_key=self.ssh_key, username=self.username)
    self.opened_ssh_client.append(ssh_client)
    return ssh_client

  def reuse_ssh_client(self):
    assert self.hostname is not None
    if not hasattr(self, 'ssh_client') or self.ssh_client == None:
      self.ssh_client = util.SshToHost(self.hostname, ssh_key=self.ssh_key, username=self.username)
    return self.ssh_client

  def CleanSshClient(self):
    if hasattr(self, 'ssh_client'):
      self.opened_ssh_client.append(self.ssh_client)
    for ssh_client in self.opened_ssh_client:
      try:
        ssh_client.close()
      except:
        pass
    self.opened_ssh_client = []
    self.ssh_client = None

  @property
  def state(self):
    return self.aws_instance.state.get('Name', None)

  def SetNameTag(self, name='tf'):
    ec2 = boto3.resource('ec2')
    self.tag = ec2.create_tags(
        Resources=[self.aws_instance.id], Tags=[{
            'Key': 'Name',
            'Value': name
        }])

  def SetCustomTag(self, key, value):
    ec2 = boto3.resource('ec2')
    self.tag = ec2.create_tags(
        Resources=[self.aws_instance.id], Tags=[{
            'Key': key,
            'Value': value
        }])

  def Start(self):
    self.aws_instance.start()

  def Stop(self):
    self.CleanSshClient()
    self.aws_instance.stop()

  def StopAndWaitUntilStopped(self):
    self.Stop()
    self.aws_instance.wait_until_stopped()

  def Terminate(self):
    self.CleanSshClient()
    self.aws_instance.terminate()

  def TerminateAndWaitUntilTerminated(self):
    self.Terminate()
    self.aws_instance.wait_until_terminated()

  @property
  def instance_id(self):
    return self.aws_instance.instance_id

  def ExecuteCommandAndWait(self, cmd, print_error=False):
    util.ExecuteCommandAndWait(
        self.reuse_ssh_client(), cmd, print_error=print_error)

  def ExecuteCommandAndReturnStdout(self, cmd):
    return util.ExecuteCommandAndReturnStdout(self.reuse_ssh_client(), cmd)

  # TODO: rename to stdout_fn
  def ExecuteCommandAndStreamOutput(self, 
                                    cmd,
                                    stdout_file=None,
                                    stderr_file=None,
                                    line_extractor=None,
                                    print_error=False,
                                    ok_exit_status=[0]):
  
    return util.ExecuteCommandAndStreamOutput(self.reuse_ssh_client(),
                                              cmd,
                                              stdout_file=stdout_file,
                                              stderr_file=stderr_file,
                                              line_extractor=line_extractor,
                                              print_error=print_error,
                                              ok_exit_status=ok_exit_status) 

  def ExecuteCommandInThread(self,
                             command,
                             stdout_file=None,
                             stderr_file=None,
                             line_extractor=None,
                             print_error=False):
    ssh_client = self.CreateSshClient()
    return util.ExecuteCommandInThread(
        ssh_client,
        command,
        stdout_file=stdout_file,
        stderr_file=stderr_file,
        line_extractor=line_extractor,
        print_error=print_error)

  def RetrieveFile(self, remote_file, local_file):
    sftp_client = self.reuse_ssh_client().open_sftp()
    sftp_client.get(remote_file, local_file)
    sftp_client.close()

  def UploadFile(self, local_file, remote_file):
    sftp_client = self.reuse_ssh_client().open_sftp()
    sftp_client.put(local_file, remote_file)
    sftp_client.close()

def CreateAwsInstances(num_instances=1,
                       image_id='',
                       instance_type='t1.micro',
                       key_name='',
                       ssh_key='',
                       instance_tag='tf',
                       security_group='default',
                       placement_group='',
                       tags=None):
  ec2 = boto3.resource('ec2')
  if placement_group:
    MaybeCreatePlacementGroup(name=placement_group)
    aws_instances = ec2.create_instances(
        ImageId=image_id,
        InstanceType=instance_type,
        MinCount=num_instances,
        MaxCount=num_instances,
        SecurityGroups=[security_group],
        Placement={'GroupName': placement_group},
        KeyName=key_name)
  else:
    aws_instances = ec2.create_instances(
        ImageId=image_id,
        InstanceType=instance_type,
        MinCount=num_instances,
        MaxCount=num_instances,
        SecurityGroups=[security_group],
        KeyName=key_name)
  assert len(aws_instances) == num_instances
  print('{} Instances created'.format(len(aws_instances)))
  instances = [
      AWSInstance(instance, ssh_key, instance_tag, tags=tags) for instance in aws_instances
  ]

  return instances


def setup_local_logdir(run):
  logdir = LOCAL_LOGDIR_PREFIX + '/' + run
  os.system('rm -Rf '+logdir)
  os.system('mkdir -p '+logdir)
  return logdir

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


def cnn_launcher():
  """Experiment launcher."""

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

  ps_job = aws.tf_job(FLAGS.run+'/ps', FLAGS.num_ps,
                      instance_type=FLAGS.ps_type,
                      placement_group=placement_group)
  worker_job = aws.tf_job(FLAGS.run+'/worker', FLAGS.num_workers,
                          instance_type=FLAGS.worker_type,
                          placement_group=placement_group)

  ps_job.wait_until_ready()
  worker_job.wait_until_ready()

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

  # kill previous running processes in case we are reusing instances
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
    task.upload("variable_mgr.py",
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
    task.upload("variable_mgr.py",
                "/home/ubuntu/Dropbox/git0/benchmarks/scripts/tf_cnn_benchmarks/variable_mgr.py")
    cmds.append(worker_cmd_tmpl % {"worker_hosts": worker_hosts_str,
                                   "ps_hosts": ps_hosts_str,
                                   "job_name": task_type,
                                   "task_index": task.id})
    task.run(' && '.join(cmds))
    print("To see the output of %s: tail -f %s" %(task.id,
                                                  task.last_stdout))


def main():
  os.system('rm -Rf data') # todo: remove
  cnn_launcher()
  

if __name__=='__main__':
  main()
  
