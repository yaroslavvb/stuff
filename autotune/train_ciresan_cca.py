# Train Ciresan's 6-layer deep MNIST network
# (from http://yann.lecun.com/exdb/mnist/)

import argparse
import os
import sys
import time
from typing import Callable, List

import autograd_lib
import globals as gl
# import torch
import scipy
import torch
import torch.nn as nn
import torchcontrib
import wandb
from attrdict import AttrDefault
from torch.utils.tensorboard import SummaryWriter

import util as u

import os
import argparse
from importlib import import_module
import shutil
import json

import torch
import torch.nn.functional as F
import wandb

# for line profiling
try:
    # noinspection PyUnboundLocalVariable
    profile  # throws an exception when profile isn't defined
except NameError:
    profile = lambda x: x  # if it's not defined simply ignore the decorator.


def main():

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--wandb', type=int, default=0, help='log to weights and biases')
    parser.add_argument('--autograd_check', type=int, default=0, help='autograd correctness checks')
    parser.add_argument('--logdir', type=str, default='/tmp/runs/curv_train_tiny/run')

    parser.add_argument('--nonlin', type=int, default=1, help="whether to add ReLU nonlinearity between layers")
    parser.add_argument('--bias', type=int, default=1, help="whether to add bias between layers")

    parser.add_argument('--layer', type=int, default=-1, help="restrict updates to this layer")
    parser.add_argument('--data_width', type=int, default=28)
    parser.add_argument('--targets_width', type=int, default=28)
    parser.add_argument('--hess_samples', type=int, default=1, help='number of samples when sub-sampling outputs, 0 for exact hessian')
    parser.add_argument('--hess_kfac', type=int, default=0, help='whether to use KFAC approximation for hessian')
    parser.add_argument('--compute_rho', type=int, default=0, help='use expensive method to compute rho')
    parser.add_argument('--skip_stats', type=int, default=1, help='skip all stats collection')

    parser.add_argument('--dataset_size', type=int, default=60000)
    parser.add_argument('--train_steps', type=int, default=1000, help="this many train steps between stat collection")
    parser.add_argument('--stats_steps', type=int, default=1000000, help="total number of curvature stats collections")

    parser.add_argument('--full_batch', type=int, default=0, help='do stats on the whole dataset')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=2e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--dropout', type=int, default=0)
    parser.add_argument('--swa', type=int, default=0)
    parser.add_argument('--lmb', type=float, default=1e-3)

    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--stats_batch_size', type=int, default=10000)
    parser.add_argument('--uniform', type=int, default=0, help='use uniform architecture (all layers same size)')
    parser.add_argument('--run_name', type=str, default='noname')

    gl.args = parser.parse_args()
    args = gl.args
    u.seed_random(1)

    gl.project_name = 'train_ciresan'
    u.setup_logdir_and_event_writer(args.run_name)
    print(f"Logging to {gl.logdir}")

    d1 = 28*28
    if args.uniform:
        d = [784, 784, 784, 784, 784, 784, 10]
    else:
        d = [784, 2500, 2000, 1500, 1000, 500, 10]
    o = 10
    n = args.stats_batch_size
    model = u.SimpleFullyConnected2(d, nonlin=args.nonlin, bias=args.bias, dropout=args.dropout)
    model = model.to(gl.device)


    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    dataset = u.TinyMNIST(data_width=args.data_width, targets_width=args.targets_width, original_targets=True,
                          dataset_size=args.dataset_size)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    train_iter = u.infinite_iter(train_loader)

    assert not args.full_batch, "fixme: validation still uses stats_iter"
    if not args.full_batch:
        stats_loader = torch.utils.data.DataLoader(dataset, batch_size=args.stats_batch_size, shuffle=False, drop_last=True)
        stats_iter = u.infinite_iter(stats_loader)
    else:
        stats_iter = None

    test_dataset = u.TinyMNIST(data_width=args.data_width, targets_width=args.targets_width, train=False, original_targets=True,
                               dataset_size=args.dataset_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.stats_batch_size, shuffle=False, drop_last=False)

    loss_fn = torch.nn.CrossEntropyLoss()
    autograd_lib.add_hooks(model)
    autograd_lib.disable_hooks()

    gl.token_count = 0
    last_outer = 0
    for step in range(args.stats_steps):
        epoch = gl.token_count // 60000
        print(gl.token_count)
        if last_outer:
            u.log_scalars({"time/outer": 1000*(time.perf_counter() - last_outer)})
        last_outer = time.perf_counter()

        # compute validation loss
        if args.swa:
            model.eval()
            with u.timeit('swa'):
                base_opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
                opt = torchcontrib.optim.SWA(base_opt, swa_start=0, swa_freq=1, swa_lr=args.lr)
                for _ in range(100):
                    optimizer.zero_grad()
                    data, targets = next(train_iter)
                    model.zero_grad()
                    output = model(data)
                    loss = loss_fn(output, targets)
                    loss.backward()
                    opt.step()
                opt.swap_swa_sgd()

        with u.timeit("validate"):
            val_accuracy, val_loss = validate(model, test_loader, f'test (epoch {epoch})')
            train_accuracy, train_loss = validate(model, stats_loader, f'train (epoch {epoch})')

        # save log
        metrics = {'epoch': epoch, 'val_accuracy': val_accuracy, 'val_loss': val_loss,
                   'train_loss': train_loss, 'train_accuracy': train_accuracy,
                   'lr': optimizer.param_groups[0]['lr'],
                   'momentum': optimizer.param_groups[0].get('momentum', 0)}
        u.log_scalars(metrics)

        # compute stats
        if args.full_batch:
            data, targets = dataset.data, dataset.targets
        else:
            data, targets = next(stats_iter)

        if not args.skip_stats:
            autograd_lib.enable_hooks()
            autograd_lib.clear_backprops(model)
            autograd_lib.clear_hess_backprops(model)
            with u.timeit("backprop_g"):
                output = model(data)
                loss = loss_fn(output, targets)
                loss.backward(retain_graph=True)
            with u.timeit("backprop_H"):
                autograd_lib.backprop_hess(output, hess_type='CrossEntropy')
            autograd_lib.disable_hooks()   # TODO(y): use remove_hooks

            with u.timeit("compute_grad1"):
                autograd_lib.compute_grad1(model)
            with u.timeit("compute_hess"):
                autograd_lib.compute_hess(model, method='kron', attr_name='hess2')
            autograd_lib.compute_stats_factored(model)

        for (i, layer) in enumerate(model.layers):
            param_names = {layer.weight: "weight", layer.bias: "bias"}
            for param in [layer.weight, layer.bias]:

                if param is None:
                    continue

                if not hasattr(param, 'stats'):
                    continue
                s = param.stats
                param_name = param_names[param]
                u.log_scalars(u.nest_stats(f"{param_name}", s))

        # gradient steps
        model.train()
        last_inner = 0
        for i in range(args.train_steps):
            if last_inner:
                u.log_scalars({"time/inner": 1000*(time.perf_counter() - last_inner)})
            last_inner = time.perf_counter()

            optimizer.zero_grad()
            data, targets = next(train_iter)
            model.zero_grad()
            output = model(data)
            loss = loss_fn(output, targets)
            loss.backward()

            optimizer.step()
            if args.weight_decay:
                for group in optimizer.param_groups:
                    for param in group['params']:
                        param.data.mul_(1-args.weight_decay)

            gl.token_count += data.shape[0]

    gl.event_writer.close()


def validate(model, val_loader, tag='validation'):
    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in val_loader:

            data, target = data.to(gl.device), target.to(gl.device)

            output = model(data)
            val_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = 100. * correct / len(val_loader.dataset)

    # TODO(y) log scalar here
    print(f'Eval: Average {tag} loss: {val_loss:.4f}, Accuracy: {correct:.0f}/{len(val_loader.dataset)} ({val_accuracy:.2f}%)')

    return val_accuracy, val_loss


if __name__ == '__main__':
    main()
