"""
(pytorch_p36) [ec2-user@ip-172-31-6-232 cifar]$ python pytorch_benchmark.py
MKL version b'Intel(R) Math Kernel Library Version 2019.0.4 Product Build 20190411 for Intel(R) 64 architecture applications'
PyTorch version 1.1.0
Scipy version:  1.3.0
Numpy version:  1.16.4
Benchmarking 1024-by-1024 matrix on cuda:0
  882.84   svd
   17.22   inv
  227.04   pinv
  452.77   eig
  227.18   svd


Laptop

MKL version unknown
PyTorch version 1.2.0
Scipy version:  1.2.1
Numpy version:  1.16.4
CPU version:  unknown
CPU logical cores: 8
CPU physical cores: 4
CPU physical sockets:  0
Benchmarking 1024-by-1024 matrix on cpu
  170.24   svd
   22.41   inv
  206.70   pinv
  247.92   eig
  180.16   pinverse
   20.08   solve
  124.89   svd
   14.57   inv
  197.24   pinv
  221.06   eig
  213.46   pinverse
   21.75   solve

"""
import os
import sys
import time

import numpy as np

import util as u

import torch

# from @eamartin
def empty_aligned(n, align):
    """Get n bytes of memory wih alignment align."""
    a = np.empty(n + (align - 1), dtype=np.float32)
    data_align = a.ctypes.data % align
    offset = 0 if data_align == 0 else (align - data_align)
    return a[offset: offset + n]


def benchmark(method):

    start_time = time.time()
    times = []

    for i in range(1):
        if method == 'svd':
            _result = torch.svd(H)
            open('/dev/null', 'w').write(str(_result[0]))
        elif method == 'inv':
            _result = torch.inverse(H)
            open('/dev/null', 'w').write(str(_result[0]))
        elif method == 'pinv':
            _result = u.pinv(H)
            open('/dev/null', 'w').write(str(_result[0]))
        elif method == 'pinverse':
            _result = torch.pinverse(H)
            open('/dev/null', 'w').write(str(_result[0]))
        elif method == 'eig':
            _result = torch.symeig(H, eigenvectors=True)
            open('/dev/null', 'w').write(str(_result[0]))
        elif method == 'svd':
            _result = torch.svd(H)
            open('/dev/null', 'w').write(str(_result[0]))
        elif method == 'solve':
            _result = torch.solve(S, H)
            open('/dev/null', 'w').write(str(_result[0]))
        else:
            assert False
        new_time = time.time()
        elapsed_time = 1000 * (new_time - start_time)
        print(f"{elapsed_time:8.2f}   {method}")
        start_time = new_time
        times.append(elapsed_time)


if __name__ == '__main__':
    methods = ['svd', 'inv', 'pinv', 'eig', 'pinverse', 'solve']*2

    u.print_version_info()
    d = 1024

    x0 = torch.rand(d).reshape((d, 1)).float()

    X = torch.rand((d, 10000))
    Y = torch.rand((d, 10000))
    H = X @ X.t()
    S = Y @ Y.t()

    if torch.cuda.is_available():
        [x0, X, Y, H, S] = u.move_to_gpu([x0, X, Y, H, S])

    print(f"Benchmarking {d}-by-{d} matrix on {x0.device}")
    for method in methods:
        benchmark(method)

# Other timings: svd
# n=1000 Times: min: 126.04, median: 132.48
# n=2000 Times: min: 573.03, median: 621.49
# n=4096 Times: min: 5586.02, median: 6032.16
# Other timings: inv
# Times: min: 17.87, median: 23.41, mean: 27.90
