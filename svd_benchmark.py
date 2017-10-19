# Fastest way to compute eigenvectors for 4k matrix?
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


from scipy import linalg  # for svd
import numpy as np
import time
import sys

methods = ['gesdd', 'gesvd', 'eigh', 'inv', 'inv2', 'linsolve']

if len(sys.argv)<2:
  method = methods[0]
else:
  method = sys.argv[1]

# from @eamartin
def empty_aligned(n, align):
  """Get n bytes of memory wih alignment align."""
  a = np.empty(n + (align - 1), dtype=np.float32)
  data_align = a.ctypes.data % align
  offset = 0 if data_align == 0 else (align - data_align)
  return a[offset : offset + n]

assert method in methods

n=4096
#n=1024
x_old = np.random.randn(n*n).reshape((n,n)).astype(dtype=np.float32)
x = empty_aligned(n*n, 32).reshape((n, n))
x[:] = x_old
x = x @ x.T

x0 = np.random.randn(n).reshape((n,1)).astype(dtype=np.float32)

start_time = time.time()
times = []

print("n=%d %s "%(n, method))
for i in range(9):
  if method == 'gesdd':
    result = linalg.svd(x)
  elif method == 'gesvd':
    result = linalg.svd(x, lapack_driver='gesvd')
  elif method == 'eigh':
    result = linalg.eigh(x)
  elif method == 'inv':
    result = linalg.inv(x)
  elif method == 'inv2':
    result = linalg.inv(x, overwrite_a=True)
  elif method == 'linsolve':
    result = linalg.solve(x, x0)
  else:
    assert False
  new_time = time.time()
  elapsed_time = 1000*(new_time - start_time)
  print("%.2f msec" %(elapsed_time))
  start_time = new_time
  times.append(elapsed_time)

print("Times: min: %.2f, median: %.2f, mean: %.2f"%(np.min(times), np.median(times), np.mean(times)))


# Other timings: svd
# n=1000 Times: min: 126.04, median: 132.48
# n=2000 Times: min: 573.03, median: 621.49
# n=4096 Times: min: 5586.02, median: 6032.16
# Other timings: inv
# Times: min: 17.87, median: 23.41, mean: 27.90
