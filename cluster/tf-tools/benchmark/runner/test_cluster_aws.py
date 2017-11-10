from command_builder import *
from pprint import pprint as pp
import yaml
import cluster_aws

from collections import OrderedDict
import time

AMI='ami-60df1418'   # cuda 8
AMI='ami-9ddb0fe5'   # boyd base
KEY_NAME='yaroslav'
KEY_FILE=os.environ['HOME']+'/d/yaroslav.pem'
SECURITY_GROUP='open'
#INSTANCE_TYPE='g3.16xlarge'
INSTANCE_TYPE='p2.8xlarge'
TAG='tf'

global_timeit_dict = OrderedDict()
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

def test_two_machine():
  


def main():
  FIRST_TIME = False
  
  if FIRST_TIME:
    with timeit('create_instances'):
      instances = cluster_aws.CreateAwsInstances(num_instances=2,
                                                 image_id=AMI,
                                                 key_name=KEY_NAME,
                                                 ssh_key=KEY_FILE,
                                                 security_group=SECURITY_GROUP,
                                                 instance_tag=TAG,
                                                 placement_group='',
                                                 instance_type=INSTANCE_TYPE)
  else:
    instances = cluster_aws.LookupAwsInstances(instance_tag=TAG,
                                               ssh_key=KEY_FILE)
    #    Exception connecting to host via ssh (could be a timeout):


  
  with timeit('connect'):
    instance = instances[0]
    instance.WaitUntilReady()
    

  def line_extractor(line):
    return True
  
  instance.ExecuteCommandAndStreamOutput('mkdir 43',
                                         stdout_file='/tmp/output')
  instance.ExecuteCommandAndStreamOutput('ls', stdout_file='/tmp/output')

  import pdb; pdb.set_trace()


if __name__=='__main__':
  main()
