# To benchmark READ throughput, run the following.
#
#     python async_sgd_benchmark.py --num-workers=10 --num-parameter-servers=10 --data-size=100000000 --read
#
# 10 workers, 10 parameter servers, 100MB objects, 64 cores:
#     read throughput: reaches 29GB/s (after warm up).
#
# 1 worker, 1 parameter server, 100MB objects, 64 cores:
#     read throughput: reaches 3GB/s
# ------------------------------------------------------------------------------
#
# To benchmark WRITE throughput, run the following.
#
#     python async_sgd_benchmark.py --num-workers=10 --num-parameter-servers=10 --data-size=100000000
#
# 10 workers, 10 parameter servers, 100MB objects, 64 cores:
#     write throughput: reaches 15GB/s (after warm up).
#
# 1 worker, 1 parameter server, 100MB objects, 64 cores:
#     write throughput: 1GB/s

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import time

import ray

parser = argparse.ArgumentParser(description='Benchmark asynchronous '
                                             'training.')
parser.add_argument('--num-workers', default=4, type=int,
                    help='The number of workers to use.')
parser.add_argument('--num-parameter-servers', default=4, type=int,
                    help='The number of parameter servers to use.')
parser.add_argument('--data-size', default=1000000, type=int,
                    help='The size of the data to use.')
parser.add_argument('--read', action='store_true',
                    help='measure read throughput, the default is to measure '
                         'write throughput')


args = parser.parse_args()

@ray.remote
class ParameterServer(object):
    def __init__(self, data_size, read):
        self.data_size = data_size
        self.read = read
        self.value = np.zeros(data_size, dtype=np.uint8)
        self.times = []

    def push(self, value):
        if not self.read:
            self.update_times()
        self.value += value

    def pull(self):
        if self.read:
            self.update_times()
        return self.value

    def update_times(self):
        self.times.append(time.time())
        if len(self.times) > 100:
            self.times = self.times[-100:]

    def get_throughput(self):
        return (self.data_size * (len(self.times) - 1) /
                (self.times[-1] - self.times[0]))


@ray.remote
def worker_task(data_size, read, *parameter_servers):
    while True:
        if read:
            # Get the current value from the parameter server.
            values = ray.get([ps.pull.remote() for ps in parameter_servers])
        else:
            # Push an update to the parameter server.
            ray.get([ps.push.remote(np.zeros(data_size, dtype=np.uint8))
                     for ps in parameter_servers])


ray.init(num_workers=args.num_workers, redirect_output=True)

parameter_servers = [ParameterServer.remote(args.data_size, args.read)
                     for _ in range(args.num_parameter_servers)]

# Let the parameter servers start up.
time.sleep(3)

# Start some training tasks.
worker_tasks = [worker_task.remote(args.data_size, args.read,
                                   *parameter_servers)
                for _ in range(args.num_workers)]

while True:
    time.sleep(2)
    throughput = (ray.get(parameter_servers[0].get_throughput.remote()) *
                  args.num_parameter_servers)
    if args.read:
        print('Read throughput is {}MB/s.'.format(throughput / 1e6))
    else:
        print('Write throughput is {}MB/s.'.format(throughput / 1e6))
