#!/usr/bin/env python
# Launch single instance

from collections import OrderedDict
from pprint import pprint as pp
import argparse
import boto3
import os
import sys
import time
#import util as myutil
import yaml

from myutil import timeit

module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, module_path+'/tf-tools/benchmark/runner')
import cluster_aws

# TODO: cluster_aws util shadows our util



EXPNAME='exp3' # TODO: remove

parser = argparse.ArgumentParser(description='ImageNet experiment')
parser.add_argument('--name', type=str, default='default',
                     help=('tag used to keep track of machines in this '
                           'experiment'))
parser.add_argument('--launch', action='store_true', default=False,
                    help='launch new instances (instead of reusing)')
parser.add_argument('--kill', action='store_true', default=False,
                    help='kill instances')
parser.add_argument('--ami', type=str, default='ami-9ddb0fe5',
                    help='AMI to use for launching instances')
parser.add_argument('--key-name', type=str, default='yaroslav',
                    help='AWS key-name to use')
parser.add_argument('--key-path', type=str,
                    default=os.environ['HOME']+'/d/yaroslav.pem',
                    help='location of .pem file on local filesystem')
parser.add_argument('--instance-type', type=str,
                     default='t2.micro',
                     help='instance type to use')
parser.add_argument('--security-group', type=str,
                     default='open',
                     help='which security group to use for instances')
parser.add_argument('--localdir_prefix', type=str, default='/temp/logs',
                     help='where to mirror worker logs locally')
parser.add_argument('--logdir_prefix', type=str, default='/efs/logs',
                     help='where to dump EFS logs')
parser.add_argument('--port', type=int, default=3333,
                     help='default port to use for all connections')

args = parser.parse_args()

def main():
  localdir=args.localdir_prefix+'/'+args.name
  logdir=args.logdir_prefix+'/'+args.name
    
  os.system('rm -Rf '+localdir)
  os.system('mkdir -p '+localdir)

  # TODO: automatically decide whether to launch or connect to existing
  # TODO: implement killing
  if args.launch:
    print("Creating new instances")
    tags = {'iam': os.environ['USER']}
    with timeit('create_instances'):
      instances = cluster_aws.CreateAwsInstances(num_instances=1,
                                                 image_id=args.ami,
                                                 key_name=args.key_name,
                                                 ssh_key=args.key_path,
                                                 security_group=args.security_group,
                                                 instance_tag=args.name,
                                                 placement_group='',
                                                 instance_type=args.instance_type,
                                                 tags=tags)
  else:
    print("Reusing existing instances")
    instances = cluster_aws.LookupAwsInstances(instance_tag=args.name,
                                               ssh_key=args.key_path)
  assert len(instances) == 1, "%d instances found" % (len(instances),)

  with timeit('connect'):
    for i,instance in enumerate(instances):
      print("Connecting to instance %d, %s" % (i, instance.instance_id))
      instance.WaitUntilReady()

  instance = instances[0]

  # TODO: mount at /efs instead of ~/efs
  setup_cmd = """
sudo apt-get install nfs-common -y
EFS_ID=fs-ab2b8102
EFS_REGION=us-west-2
sudo mkdir -p /efs
sudo mount -t nfs -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 $EFS_ID.efs.$EFS_REGION.amazonaws.com:/ /efs"""

  setup_cmds = setup_cmd.strip().split('\n')
  cmd = ' && '.join(setup_cmds)
  i = 0
  fn_out = localdir + '/out-%02d'%(i,)
  fn_err = localdir + '/err-%02d'%(i,)

  print(cmd)
  def p(line): print(line)
  instance.ExecuteCommandAndStreamOutput(cmd, fn_out, fn_err, p)

  
if __name__=='__main__':
  main()
