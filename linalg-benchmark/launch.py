#!/usr/bin/env python
# Run linalg benchmark on AWS

import argparse
import ncluster
ncluster.set_backend('aws')

import threading

parser = argparse.ArgumentParser(description='launch')
parser.add_argument('--instances', default='p3.16xlarge, c5.18xlarge, c5.9xlarge, m5.24xlarge, i3.metal, g3.16xlarge')
parser.add_argument('--image', default="Deep Learning AMI (Amazon Linux) Version 13.0")
parser.add_argument('--N', default='')
parser.add_argument('--short', action='store_true', help='short version of benchmark')
args = parser.parse_args()

results = {}
def launch(instance):
  """Run benchmark on given instance type."""
  task = ncluster.make_task('benchmark-'+instance, instance_type=instance, image_name=args.image)
  task.upload('benchmark.py')
  task.run('source activate tensorflow_p36')
  task.run('pip install torch')
  task.run('export CUDA_VISIBLE_DEVICES=0')
  if args.N:
    task.run(f'export LINALG_BENCHMARK_N={args.N}')
  if args.short:
    task.run(f'export LINALG_BENCHMARK_SHORT={args.N}')
    
  stdout, stderr = task.run_with_output('python benchmark.py')
  print('='*80)
  print(instance)
  print(stdout)


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



if __name__=='__main__':
  main()
