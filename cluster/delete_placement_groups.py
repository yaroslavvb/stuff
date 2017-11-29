#!/usr/bin/env python

# delete all placement groups

import boto3

# {'PlacementGroups': [{'GroupName': 'gpu12',
#    'State': 'available',
#    'Strategy': 'cluster'},
#   {'GroupName': 'gpu6', 'State': 'available', 'Strategy': 'cluster'},
#   {'GroupName': 'gpu10', 'State': 'available', 'Strategy': 'cluster'},
#   {'GroupName': 'gpu4', 'State': 'available', 'Strategy': 'cluster'},
#   {'GroupName': 'cnn2', 'State': 'available', 'Strategy': 'cluster'},
#   {'GroupName': 'gpu5', 'State': 'available', 'Strategy': 'cluster'},
#   {'GroupName': 'gpu3', 'State': 'available', 'Strategy': 'cluster'},
#   {'GroupName': 'tf', 'State': 'available', 'Strategy': 'cluster'},
#   {'GroupName': 'gpu7', 'State': 'available', 'Strategy': 'cluster'},
#   {'GroupName': 'gpu11', 'State': 'available', 'Strategy': 'cluster'},
#   {'GroupName': 'gpu8', 'State': 'available', 'Strategy': 'cluster'},
#   {'GroupName': 'gpu9', 'State': 'available', 'Strategy': 'cluster'},
#   {'GroupName': 'cnn', 'State': 'available', 'Strategy': 'cluster'}],
#  'ResponseMetadata': {'HTTPHeaders': {'content-type': 'text/xml;charset=UTF-8',
#    'date': 'Tue, 28 Nov 2017 18:52:18 GMT',
#    'server': 'AmazonEC2',
#    'transfer-encoding': 'chunked',
#    'vary': 'Accept-Encoding'},
#   'HTTPStatusCode': 200,
#   'RequestId': '3d7adfe7-1109-413d-9aab-2f0aeafef968',
#   'RetryAttempts': 0}}

import boto3
ec2 = boto3.client('ec2')

result=ec2.describe_placement_groups()
#print(result)
for entry in result["PlacementGroups"]:
  name = entry.get('GroupName', '---')
  try:
    print("Deleting "+name)
    response = ec2.delete_placement_group(GroupName=name)
    print("Response was %d" %(response['ResponseMetadata']['HTTPStatusCode']))
  except Exception as e:
    print("Failed with %s"%(e,))
