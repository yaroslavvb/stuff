# compare timing for variety of batch-sizes
# TODO: make PyTorch not run out of memory

import tensorflow as tf
import eager_lbfgs
import pytorch_lbfgs
import numpy as np
import util as u

import time
import sys
import os

def run_experiment(iters, name):

  #batch_sizes = [1, 10, 100, 1000, 10000, 60000]
  batch_sizes = [100, 200, 300]

  eager_stats = []
  pytorch_stats = []

  def benchmark(f):
    # do whole run once for pre-warming
    f()
    import gc; gc.collect()
    start_time = time.perf_counter()
    final_loss = f()
    elapsed_time = time.perf_counter() - start_time
    return final_loss, elapsed_time

  for batch_size in batch_sizes:
    def eager_run():
      return eager_lbfgs.benchmark(batch_size=batch_size, iters=iters)
    eager_stats.append(benchmark(eager_run))
    def pytorch_run():
      return pytorch_lbfgs.benchmark(batch_size=batch_size, iters=iters)
    pytorch_stats.append(benchmark(pytorch_run))

  print(eager_stats)
  print(pytorch_stats)
  # pytorch_losses
  # pytorch_times
  # pytorch_sizes
  
  eager_stats = np.array(eager_stats)
  pytorch_stats = np.array(pytorch_stats)
  u.dump(batch_sizes, name+"_batch.csv")
  
  u.dump(eager_stats[:,0], name+"_eager_loss.csv")
  u.dump(eager_stats[:,1], name+"_eager_time.csv")
  
  u.dump(pytorch_stats[:,0], name+"_pytorch_loss.csv")
  u.dump(pytorch_stats[:,1], name+"_pytorch_time.csv")

    
if __name__=='__main__':
  if len(sys.argv)<2:
    print("Running short comparison")
    run_experiment(51, "short")
  else:
    print("Running long comparison")
    run_experiment(101, "long")
    
