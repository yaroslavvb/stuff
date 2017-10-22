#!/bin/env python
import kfac_pytorch as kfac_lib
import util as u
import torch.nn.functional as F

nonlin = F.relu

def run_experiment(name, optimizer, nonlin, kfac, iters, lr):
  losses, vlosses = kfac_lib.train(optimizer=optimizer, nonlin=nonlin,
                                   kfac=kfac, iters=iters, lr=lr)
  u.dump(losses, 'losses_'+name+'.csv', True)
  u.dump(vlosses, 'vlosses_'+name+'.csv', True)

run_experiment('sgd', 'sgd', F.sigmoid, False, 1000, 0.2)
run_experiment('adam', 'adam', F.sigmoid, False, 1000, 1e-3)
run_experiment('sgd_kfac', 'sgd', F.sigmoid, True, 200, 0.2)
run_experiment('adam_kfac', 'adam', F.sigmoid, True, 200, 1e-3)

# relu explodes, divide learning rates by 10
run_experiment('sgd2', 'sgd', F.relu, False, 1000, 0.02)
run_experiment('adam2', 'adam', F.relu, False, 1000, 1e-4)
run_experiment('sgd_kfac2', 'sgd', F.relu, True, 200, 0.02)
run_experiment('adam_kfac2', 'adam', F.relu, True, 200, 1e-4)
