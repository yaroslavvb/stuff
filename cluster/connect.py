#!/usr/bin/env python
"""

Script to connect to most recent instance with containing given fragment:
Usage:
connect
-- connects to most recently launched instance
connect i3
-- connects to most recently launchedn instance containing i3 in instance id


Debugging/exploring:

python
from pprint import pprint
import boto3
ec2 = boto3.client('ec2')
response = ec2.describe_instances()
reservation=response['Reservations'][0]
instance = reservation['Instances'][0]
pprint(instance)
"""

# todo: allow to do ls, show tags
# todo: handle KeyError: 'PublicIpAddress'

import boto3
import time
import sys
import os
from datetime import datetime
from operator import itemgetter


def toseconds(dt):
  # to invert:
  # import pytz
  # utc = pytz.UTC
  # utc.localize(datetime.fromtimestamp(seconds))
  return time.mktime(dt.utctimetuple())

def main():
  fragment = ''
  if len(sys.argv)>1:
    fragment = sys.argv[1]
    
  ec2 = boto3.client('ec2')
  response = ec2.describe_instances()

  instance_list = []
  for reservation in response['Reservations']:
    for instance in reservation['Instances']:
      instance_list.append((toseconds(instance['LaunchTime']), instance))

  import pytz
  from tzlocal import get_localzone # $ pip install tzlocal

  sorted_instance_list = sorted(instance_list, key=itemgetter(0))
  cmd = ''
  for (ts, instance) in reversed(sorted_instance_list):
    if fragment in instance['InstanceId']:
      
      localtime = instance['LaunchTime'].astimezone(get_localzone())
      keyname = instance.get('KeyName','none')
      print("Connecting to %s launched at %s with key %s" % (instance['InstanceId'], localtime, keyname))
      cmd = "ssh -i $HOME/Dropbox/yaroslav.pem -o StrictHostKeyChecking=no ubuntu@"+instance['PublicIpAddress']
      break
  if not cmd:
    print("no instance id contains fragment '%s'"%(fragment,))
  else:
    print(cmd)
    os.system(cmd)



if __name__=='__main__':
  main()
