# launch imagenet experiment

from command_builder import *
from pprint import pprint as pp
import yaml
import cluster_aws
import util
import boto3

from collections import OrderedDict
import time

import argparse
parser = argparse.ArgumentParser(description='ImageNet experiment')

parser.add_argument('--launch', action='store_true', default=False,
                    help='launch new instances (instead of reusing)')
parser.add_argument('--num-workers', type=int, default=1,
                    help='number of workers')
parser.add_argument('--num-ps', type=int, default=1,
                    help='number of parameter servers')
parser.add_argument('--ami', type=str, default='ami-9ddb0fe5',
                    help='AMI to use for launching instances')
parser.add_argument('--key-name', type=str, default='yaroslav',
                    help='AWS key-name to use')
parser.add_argument('--key-path', type=str,
                    default=os.environ['HOME']+'/d/yaroslav.pem',
                    help='location of .pem file on local filesystem')
parser.add_argument('--instance-type', type=str,
                     default='p2.8xlarge',
                     help='instance type to use')
parser.add_argument('--security-group', type=str,
                     default='open',
                     help='which security group to use for instances')
parser.add_argument('--tag', type=str, default='tf',
                     help=('tag used to keep track of machines in this '
                           'experiment'))
parser.add_argument('--logdir', type=str, default='/tmp/tf',
                     help='where to dump worker logs')
parser.add_argument('--port', type=int, default=3333,
                     help='default port to use for all connections')

args = parser.parse_args()


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


def main():
  num_instances = args.num_workers + args.num_ps
  os.system('rm -Rf '+args.logdir)
  os.system('mkdir -p '+args.logdir)

  # TODO: add these commands (running manually for now)
  #  sudo nvidia-persistenced
  #  sudo nvidia-smi --auto-boost-default=0
  #  sudo nvidia-smi -ac 2505,875 # p2
  
  if args.launch:
    print("Creating new instances")
    with timeit('create_instances'):
      instances = cluster_aws.CreateAwsInstances(num_instances=num_instances,
                                                 image_id=args.ami,
                                                 key_name=args.key_name,
                                                 ssh_key=args.key_path,
                                                 security_group=args.security_group,
                                                 instance_tag=args.tag,
                                                 placement_group='',
                                                 instance_type=args.instance_type)
  else:
    # TODO: better control of retrieved instances
    print("Reusing existing instances")
    instances = cluster_aws.LookupAwsInstances(instance_tag=args.tag,
                                               ssh_key=args.key_path)
    assert len(instances) >= num_instances

  # todo: deterministic worker sort
  with timeit('connect'):
    for i,instance in enumerate(instances):
      if i >= num_instances:
        break
      print("Connecting to instance %d, %s" % (i, instance.instance_id))
      instance.WaitUntilReady()


  worker_instances = instances[:args.num_workers]
  ps_instances = instances[args.num_workers:args.num_workers+args.num_ps]

  instance_ip_map = get_instance_ip_map()
  
  worker_host_fragments = []
  for instance in worker_instances:
    assert instance.instance_id in instance_ip_map
    worker_host_str = '%s:%d'%(instance_ip_map[instance.instance_id], args.port)
    worker_host_fragments.append(worker_host_str)
  worker_hosts = ','.join(worker_host_fragments)
  
  ps_host_fragments = []
  for instance in ps_instances:
    assert instance.instance_id in instance_ip_map
    ps_host_str = '%s:%d'%(instance_ip_map[instance.instance_id], args.port)
    ps_host_fragments.append(ps_host_str)
  ps_hosts = ','.join(ps_host_fragments)


  line_extractor = util.ExtractErrorToConsole

  setup_cmd = "source ~/.bashrc && export PATH=~/anaconda3/bin:$PATH && source activate py2 && cd ~/git0/benchmarks/scripts/tf_cnn_benchmarks"
  
  worker_cmd_tmpl = "python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=64 --num_batches=1000 --model=resnet50 --optimizer=sgd --variable_update=distributed_replicated --cross_replica_sync=True --local_parameter_device=gpu --num_gpus=1 --nodistortions --display_every=10 --worker_hosts=%(worker_hosts)s --ps_hosts=%(ps_hosts)s --job_name=worker --task_index=%(task_index)s"

  #  job_name = 'worker'
  for i, instance in enumerate(worker_instances):
    worker_cmd = worker_cmd_tmpl % {'worker_hosts': worker_hosts, 'ps_hosts': ps_hosts, 'task_index': i}
    cmd = setup_cmd + " && " + worker_cmd
    #    print(cmd)
    fn_out = args.logdir + '/worker_out-%02d'%(i,)
    fn_err = args.logdir + '/worker_err-%02d'%(i,)
    #ssh_client = instance.reuse_ssh_client()
    result = instance.ExecuteCommandInThread(cmd,
                                             stdout_file=fn_out,
                                             stderr_file=fn_err,
                                             line_extractor=line_extractor)
    print("worker %d started" %(i,))

  ps_cmd_tmpl = "CUDA_VISIBLE_DEVICES='' python tf_cnn_benchmarks.py --local_parameter_device=gpu --worker_hosts=%(worker_hosts)s --ps_hosts=%(ps_hosts)s --job_name=ps --task_index=%(task_index)s"
  job_name = 'ps'
  for i, instance in enumerate(ps_instances):
    ps_cmd = ps_cmd_tmpl % {'worker_hosts': worker_hosts, 'ps_hosts': ps_hosts,
                            'task_index': i}
    cmd = setup_cmd + " && " + ps_cmd
    #    print(cmd)
    fn_out = args.logdir + '/ps_out-%02d'%(i,)
    fn_err = args.logdir + '/ps_err-%02d'%(i,)
    #ssh_client = instance.reuse_ssh_client()
    result = instance.ExecuteCommandInThread(cmd,
                                             stdout_file=fn_out,
                                             stderr_file=fn_err,
                                             line_extractor=line_extractor)
    print("parameter server %d started " %(i,))

  time.sleep(10000)
  
if __name__=='__main__':
  main()
