import os
import sys
import time
from typing import Optional, Tuple, Callable

# import torch
import scipy
import torch
from torchcurv.optim import SecondOrderOptimizer


import torch.nn as nn

import util as u

import numpy as np

"""
MKL version unknown
PyTorch version 1.2.0
Scipy version:  1.2.1
Numpy version:  1.16.4
1024-by-1024 matrix
 7079.93   linalg.solve_lyapunov
  280.11   linalg.pinvh
 1186.08   linalg.pinv
   49.18   linalg.inv
  118.23   qr
  413.42   svd
"""

class Net(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1, bias=False)

    def forward(self, x: torch.Tensor):
        result = self.w(x)
        return result

last_time = 0

class timeit:
    """Decorator to measure length of time spent in the block in millis and log
    it to TensorBoard. This function is
    """

    def __init__(self, tag=""):
        self.tag = tag

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        global last_time
        self.end = time.perf_counter()
        interval_ms = 1000 * (self.end - self.start)
        torch.cuda.synchronize()
        print(f"{interval_ms:8.2f}   {self.tag}")
        last_time = interval_ms


def get_mkl_version():
  import ctypes
  import numpy as np

  # this recipe only works on Linux
  try:
    ver = np.zeros(199, dtype=np.uint8)
    mkl = ctypes.cdll.LoadLibrary("libmkl_rt.so")
    mkl.MKL_Get_Version_String(ver.ctypes.data_as(ctypes.c_char_p), 198)
    return ver[ver != 0].tostring()
  except:
    return 'unknown'


def print_cpu_info():
  ver = 'unknown'
  try:
    for l in open("/proc/cpuinfo").read().split('\n'):
      if 'model name' in l:
        ver = l
        break
  except:
    pass


def linalg_bench():
    if np.__config__.get_info("lapack_mkl_info"):
        print("MKL version", get_mkl_version())
    else:
        print("not using MKL")

    print("PyTorch version", torch.version.__version__)

    print("Scipy version: ", scipy.version.full_version)
    print("Numpy version: ", np.version.full_version)

    times = {}
    dimlist = [768, 2500, 2000, 1500, 1000, 500]
    for device in ['cpu', 'cuda']:
        for d in dimlist*2:
            print(f"{d}-by-{d} matrix: ", device)
            n = 10000
            assert n > 2*d   # to prevent singularity
            X = u.from_numpy(np.random.random((d, 10000)))
            Y = u.from_numpy(np.random.random((d, 10000)))
            H = X @ X.t()
            S = Y @ Y.t()
            if torch.cuda.is_available():
                H = H.to(device)
                S = S.to(device)

            with timeit(f"symeig"):
                result = torch.symeig(H, eigenvectors=False)
            times.setdefault(d, []).append(last_time)
            
            with timeit(f"symeig"):
                result = torch.symeig(H, eigenvectors=True)
            times.setdefault(d, []).append(last_time)

            with timeit(f"svd"):
                result = torch.svd(H)
            times.setdefault(d, []).append(last_time)
    for d in times:
        print(f"{d}-by-{d}: {u.format_list(times[d][len(times[d])//2:])}")


if __name__ == '__main__':
    linalg_bench()
