from pprint import pprint as pp
import yaml
#import util
import boto3
from collections import OrderedDict
import time

class timeit:
  """Decorator to measure length of time spent in the block in millis and log
  it to TensorBoard."""
  
  def __init__(self, tag=""):
    self.tag = tag
    
  def __enter__(self):
    self.start = time.perf_counter()
    return self
  
  def __exit__(self, *args):
    self.end = time.perf_counter()
    interval_sec = (self.end - self.start)
    print("%s took %.2f seconds"%(self.tag, interval_sec))

def get_instance_ip_map():
  """Return instance_id->private_ip map for all running instances."""
  
  ec2 = boto3.resource('ec2')

  # Get information for all running instances
  running_instances = ec2.instances.filter(Filters=[{
    'Name': 'instance-state-name',
    'Values': ['running']}])

  ec2info = OrderedDict()
  for instance in running_instances:
    name = ''
    for tag in instance.tags or []:
      if 'Name' in tag['Key']:
        name = tag['Value']
    ec2info[instance.id] = instance.private_ip_address
    
  return ec2info
