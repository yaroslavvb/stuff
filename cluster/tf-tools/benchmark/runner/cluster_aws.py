import boto3
import os
import time
import util
from contextlib import contextmanager


class AWSInstance(object):

  def __init__(self, instance, ssh_key='', name='', username='ubuntu'):
    # assert instance is aws instance
    self.aws_instance = instance
    self.ssh_key = ssh_key
    self.username = username
    if name:
      self.SetNameTag(name)
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


def MaybeCreatePlacementGroup(name='tf_bm'):
  client = boto3.client('ec2')
  try:
    client.describe_placement_groups(GroupNames=[name])
  except boto3.exceptions.botocore.exceptions.ClientError as e:
    res = client.create_placement_group(GroupName=name, Strategy='cluster')

  counter = 0
  while True:
    try:
      res = client.describe_placement_groups(GroupNames=[name])
      if res['PlacementGroups'][0]['State'] == 'available':
        break
    except:
      pass
    counter = counter + 1
    if counter >= 10:
      print('Failed to create placement group %s' % name)
    time.sleep(10)


def DeletePlacementGroup(name='tf_bm'):
  client = boto3.client('ec2')
  try:
    client.describe_placement_groups(GroupNames=[name])
  except boto3.exceptions.botocore.exceptions.ClientError as e:
    print("Placement group %s doesn't exit." % name)
    return

  # Not sure whether delete_placement_group would throw or not.
  res = client.delete_placement_group(GroupName=name)
  if res['ResponseMetadata']['HTTPStatusCode'] != 200:
    print('Failed to delete placement group %s' % name)




def CreateAwsInstances(num_instances=1,
                       image_id='',
                       instance_type='t1.micro',
                       key_name='',
                       ssh_key='',
                       instance_tag='tf',
                       security_group='default',
                       placement_group=''):
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
      AWSInstance(instance, ssh_key, instance_tag) for instance in aws_instances
  ]

  return instances


def LookupAwsInstances(image_id=None,
                       state=None,
                       instance_tag=None,
                       placement_group=None,
                       ssh_key=None):

  def FillOneFilter(key, values):
    f = {}
    f['Name'] = key
    f['Values'] = values
    return f

  filters = []
  if image_id is not None:
    filters.append(FillOneFilter('image-id', [image_id]))
  if state is not None:
    filters.append(FillOneFilter('instance-state-name', [state.lower()]))
  if instance_tag is not None:
    filters.append(FillOneFilter('tag:Name', [instance_tag]))
  if placement_group is not None:
    filters.append(FillOneFilter('placement-group-name', [placement_group]))

  ec2 = boto3.resource('ec2')
  instances = ec2.instances.filter(Filters=filters)

  return [
      AWSInstance(instance, ssh_key) for instance in instances
      if instance.state.get('Name') != 'terminated'
  ]


@contextmanager
def AwsInstances(num_instances=1,
                 image_id='',
                 instance_type='',
                 key_name='',
                 ssh_key='',
                 security_group='default',
                 instance_tag='',
                 placement_group='bm_group',
                 close_behavior=None):
  try:
    instances_created = False
    instances = CreateAwsInstances(
        num_instances=num_instances,
        image_id=image_id,
        instance_type=instance_type,
        key_name=key_name,
        ssh_key=ssh_key,
        instance_tag=instance_tag,
        security_group=security_group,
        placement_group=placement_group)
    instances_created = True

    for instance in instances:
      print('Waiting for instance({}) to be ready.'.format(
          instance.instance_id))
      instance.WaitUntilReady()
    print('All {} instances ready!!!'.format(len(instances)))
    yield instances
  finally:
    if not instances_created:
      return
    if close_behavior is not None:
      for instance in instances:
        if close_behavior == 'terminate':
          instance.Terminate()
        elif close_behavior == 'stop':
          instance.Stop()
      if placement_group and close_behavior == 'terminate':
        DeletePlacementGroup(placement_group)


@contextmanager
def ReuseAwsInstances(image_id=None,
                      state=None,
                      instance_tag=None,
                      placement_group=None,
                      ssh_key=None,
                      close_behavior=None):
  try:
    instances = LookupAwsInstances(
        image_id=image_id,
        instance_tag=instance_tag,
        state=state,
        placement_group=placement_group,
        ssh_key=ssh_key)

    if len(instances) == 0:
      raise ValueError('No instances found for instance_tag={} or image_id={}'.
                       format(instance_tag, image_id))

    for instance in instances:
      if instance.state.lower() != 'running':
        print('Current instance({}) state:{}, trying to start.'.format(
            instance.instance_id, instance.state))
        instance.Start()
    for instance in instances:
      print('Waiting for instance({}) to be ready.'.format(
          instance.instance_id))
      instance.WaitUntilReady()
    print('All {} instances ready!!!'.format(len(instances)))
    yield instances
  finally:
    if close_behavior is not None:
      for instance in instances:
        if close_behavior == 'terminate':
          instance.Terminate()
        elif close_behavior == 'stop':
          instance.Stop()
      if placement_group and close_behavior == 'terminate':
        DeletePlacementGroup(placement_group)
