#!/usr/bin/env python
"""

Script to kill all instances matching given prefix.

Usage:

./terminate_instances.py gpu   # terminates all instances matching "gpu*"
"""

import boto3
import time
import sys
import os

LIMIT_TO_KEY = 'yaroslav'   # only touch instances launched with this key,
                            # set to '' to remove restriction

def main():
  prefix = sys.argv[1]

  ec2 = boto3.client('ec2')
  response = ec2.describe_instances()

  def get_name(instance_response):
    names = [entry['Value'] for entry in instance_response.get('Tags',[]) if
             entry['Key']=='Name']
    if not names:
      names = ['']
    assert len(names)==1
    return names[0]

  instance_list = []
  for reservation in response['Reservations']:
    for instance_response in reservation['Instances']:
      instance_list.append((get_name(instance_response),
                            instance_response))

  instances_to_kill = []
  for (name, instance_response) in instance_list:
    if not name.startswith(prefix):
      continue
    key = instance_response.get('KeyName', '')
    if LIMIT_TO_KEY and LIMIT_TO_KEY != key:
      print("instance %s matches but key %s doesn't match desired key %s, "
            "skipping" %(name, key, LIMIT_TO_KEY))
      continue
    state = instance_response['State']['Name']
    if state == 'terminated':
      continue
    instances_to_kill.append((instance_response['InstanceId'],
                              name,
                              instance_response['AmiLaunchIndex'],
                              state))

  for (instance_id, name, task_id, state) in instances_to_kill:
    print("%s:%s   %s"%(name, task_id, state))

  answer = input("%d instances found, terminate? (Y/n) " % (
    len(instances_to_kill)))
  if not answer:
    answer = "y"
  if answer.lower() == "y":
    instance_ids = [record[0] for record in instances_to_kill]
    response = ec2.terminate_instances(InstanceIds=instance_ids)
    print("Terminating, got response: %s", response)
  else:
    print("Didn't get y, doing nothing")
  

if __name__=='__main__':
  main()
