#!/usr/bin/env python
# Run crashing TensorFlow SVD example

import ncluster
ncluster.set_backend('aws')

import argparse
parser = argparse.ArgumentParser(description='launch')
parser.add_argument('--instance', default='c5.9xlarge')
parser.add_argument('--image', default="Deep Learning AMI (Amazon Linux) Version 13.0")
args = parser.parse_args()

def main():
  task = ncluster.make_task(instance_type=args.instance,
                            image_name=args.image)
  task.run('source activate tensorflow_p36')
  task.upload('tensorflow_svd_crash.py')
  stdout, stderr = task.run_with_output('python tensorflow_svd_crash.py')
  print(stdout, stderr)

if __name__=='__main__':
  main()
