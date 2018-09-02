#!/usr/bin/env python
# Run linalg benchmark on AWS

import argparse
import ncluster
ncluster.set_backend('aws')

import threading

parser = argparse.ArgumentParser(description='launch')
parser.add_argument('--instances', default='p3.16xlarge, c5.18xlarge, c5.9xlarge, t3.2xlarge, m5.24xlarge, i3.metal, g3.16xlarge')
parser.add_argument('--image', default="Deep Learning AMI (Amazon Linux) Version 13.0")
args = parser.parse_args()

results = {}
def launch(instance):
  """Run benchmark on given instance type."""
  task = ncluster.make_task(instance_type=instance, image_name=args.image)
  task.upload('benchmark.py')
  task.run('source activate tensorflow_p36')
  task.run('pip install torch')
  stdout, stderr = task.run_with_output('python benchmark.py')
  results[instance] = stdout


def main():
  # launch 
  threads = []
  for instance in args.instances.split(','):
    instance = instance.strip()
    thread = threading.Thread(target=launch, args=[instance])
    thread.start()
    threads.append(thread)
  for thread in threads:
    thread.join()

  for instance_type in results:
    print(f"Results for {instance_type}")
    print(results[instance_type])


if __name__=='__main__':
  main()
