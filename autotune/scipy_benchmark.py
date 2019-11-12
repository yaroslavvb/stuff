"""
MKL version b'Intel(R) Math Kernel Library Version 2019.0.4 Product Build 20190411 for Intel(R) 64 architecture applications'
PyTorch version 1.1.0
Scipy version:  1.3.0
Numpy version:  1.16.4
Benchmarking 1024-by-1024 matrix
  246.61   gesdd
  472.90   gesvd
  160.03   eigh
   22.65   inv
   20.15   inv2
   21.93   linsolve
 1505.37   pinv
  229.98   pinv2
  173.93   pinvh
 5800.83   lyapunov
"""

import os

from scipy import linalg  # for svd
import numpy as np
import time
import sys

import util as u


# from @eamartin
def empty_aligned(n, align):
    """Get n bytes of memory wih alignment align."""
    a = np.empty(n + (align - 1), dtype=np.float32)
    data_align = a.ctypes.data % align
    offset = 0 if data_align == 0 else (align - data_align)
    return a[offset: offset + n]


def benchmark(method, d):
    #x_old = np.random.randn(d * d).reshape((d, d)).astype(dtype=np.float32)
    #x = empty_aligned(d * d, 32).reshape((d, d))
    #x[:] = x_old
    #x = x @ x.T

    x0 = np.random.randn(d).reshape((d, 1)).astype(dtype=np.float32)

    X = np.random.random((d, 10000))
    Y = np.random.random((d, 10000))
    H = X @ X.T
    S = Y @ Y.T

    Xcov = H

    start_time = time.time()
    times = []

    for i in range(1):
        if method == 'gesdd':
            _result = linalg.svd(Xcov)
        elif method == 'gesvd':
            _result = linalg.svd(Xcov, lapack_driver='gesvd')
        elif method == 'eigh':
            _result = linalg.eigh(Xcov)
        elif method == 'inv':
            _result = linalg.inv(Xcov)
        elif method == 'inv2':
            _result = linalg.inv(Xcov, overwrite_a=True)
        elif method == 'linsolve':
            _result = linalg.solve(Xcov, x0)
        elif method == 'pinv':
            _result = linalg.pinv(Xcov)
        elif method == 'pinv2':
            _result = linalg.pinv2(Xcov)
        elif method == 'pinvh':
            _result = linalg.pinvh(Xcov)
        elif method == 'lyapunov':
            _result = linalg.solve_lyapunov(H, S)
        else:
            assert False
        new_time = time.time()
        elapsed_time = 1000 * (new_time - start_time)
        print(f"{elapsed_time:8.2f}   {method}")
        start_time = new_time
        times.append(elapsed_time)


if __name__ == '__main__':
    methods = ['gesdd', 'gesvd', 'eigh', 'inv', 'inv2', 'linsolve', 'pinv', 'pinv2', 'pinvh', 'lyapunov']

    u.print_version_info()
    d = 1024
    print(f"Benchmarking {d}-by-{d} matrix")

    for method in methods:
        benchmark(method, d)

# Older timings:
# Fastest way to compute eigenvectors for 4k matrix?
#
# Inverse on i3.metal
# n=4096: 368 ms ± 1.51 ms per loop
#
# Xeon V3 benchmarks:
# n=4096 eigs  min: 27758.34, median: 28883.69
# n=4096 gesdd min: 7241.70, median: 8477.95
# n=4096 gesvd min=20487.48, median: 22057.64,
# n=4096 inv min: 556.67, median: 579.25,
# n=4096 linsolve: min: 534.40, median: 558.06, mean: 579.19
#
# Xeon V4:
# n=4096 gesdd min: 5586.02, median: 6032.16
#
#
# i7-5820K CPU @ 3.30GHz
# n=4096 gesdd 7288.02, median: 7397.23, mean: 7478.78
# n=4096 inv 520 msec
#
# after upgrading things
# b'Intel(R) Math Kernel Library Version 2017.0.3 Product Build 20170413 for Intel(R) 64 architecture applications'
# n=4096 inv 1427.54
# Other timings: svd
# n=1000 Times: min: 126.04, median: 132.48
# n=2000 Times: min: 573.03, median: 621.49
# n=4096 Times: min: 5586.02, median: 6032.16
# Other timings: inv
# Times: min: 17.87, median: 23.41, mean: 27.90
