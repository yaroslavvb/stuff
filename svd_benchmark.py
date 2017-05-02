# Fastest way to compute eigenvectors for 4k matrix?
#
# Xeon V3 benchmarks:
# n=4096 eigs  min: 27758.34, median: 28883.69
# n=4096 gesdd min: 7241.70, median: 8477.95
# n=4096 gesvd min=20487.48, median: 22057.64,
#
# Xeon V4:
# n=4096 gesdd min: 5586.02, median: 6032.16

import util
from scipy import linalg  # for svd
import numpy as np
import time
import sys

methods = ['gesdd', 'gesvd', 'eigh']

if len(sys.argv)<2:
  method = methods[0]
else:
  method = sys.argv[1]

assert method in methods

n=4096
x = np.random.randn(n*n).reshape((n,n)).astype(dtype=np.float32)
x = x @ x.T
util.record_time()
start_time = time.time()
times = []

print("n=%d %s "%(n, method))
for i in range(9):
  if method == 'gesdd':
    result = linalg.svd(x)
  elif method == 'gesvd':
    result = linalg.svd(x)
  elif method == 'eigh':
    result = linalg.eigh(x)
  else:
    assert False
  new_time = time.time()
  elapsed_time = 1000*(new_time - start_time)
  print("%.2f msec" %(elapsed_time))
  start_time = new_time
  times.append(elapsed_time)

print("Times: min: %.2f, median: %.2f, mean: %.2f"%(np.min(times), np.median(times), np.mean(times)))


# Other timings
# n=1000 Times: min: 126.04, median: 132.48
# n=2000 Times: min: 573.03, median: 621.49
# n=4096 Times: min: 5586.02, median: 6032.16
