# Train Ciresan's 6-layer deep MNIST network
# (from http://yann.lecun.com/exdb/mnist/)

import argparse
import os
import sys
import time
from typing import Callable, List

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
from torchvision import datasets, transforms, models
import torchcurv
from torchcurv.optim import SecondOrderOptimizer, VIOptimizer
from torchcurv.utils import Logger


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

    parser.add_argument('--wandb', type=int, default=1, help='log to weights and biases')
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
    parser.add_argument('--train_steps', type=int, default=5, help="this many train steps between stat collection")
    parser.add_argument('--stats_steps', type=int, default=1000000, help="total number of curvature stats collections")

    parser.add_argument('--full_batch', type=int, default=0, help='do stats on the whole dataset')
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--stats_batch_size', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--dropout', type=int, default=0)
    parser.add_argument('--swa', type=int, default=1)
    parser.add_argument('--lmb', type=float, default=1e-3)
    parser.add_argument('--uniform', type=int, default=0, help="all layers same size")
    parser.add_argument('--redundancy', type=int, default=0, help="duplicate all layers this many times")
    args = parser.parse_args()

    attemp_count = 0
    while os.path.exists(f"{args.logdir}{attemp_count:02d}"):
        attemp_count += 1
    logdir = f"{args.logdir}{attemp_count:02d}"

    run_name = os.path.basename(logdir)
    gl.event_writer = SummaryWriter(logdir)
    print(f"Logging to {run_name}")
    u.seed_random(1)

    d1 = 28*28
    if args.uniform:
        d = [784, 784, 784, 784, 784, 784, 10]
    else:
        d = [784, 2500, 2000, 1500, 1000, 500, 10]
    o = 10
    n = args.stats_batch_size
    if args.redundancy:
        model = u.RedundantFullyConnected2(d, nonlin=args.nonlin, bias=args.bias, dropout=args.dropout, redundancy=args.redundancy)
    else:
        model = u.SimpleFullyConnected2(d, nonlin=args.nonlin, bias=args.bias, dropout=args.dropout)
    model = model.to(gl.device)

    try:
        # os.environ['WANDB_SILENT'] = 'true'
        if args.wandb:
            wandb.init(project='train_ciresan', name=run_name)
            wandb.tensorboard.patch(tensorboardX=False)
            wandb.config['train_batch'] = args.train_batch_size
            wandb.config['stats_batch'] = args.stats_batch_size
            wandb.config['redundancy'] = args.redundancy
    except Exception as e:
        print(f"wandb crash with {e}")

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

    gl.token_count = 0
    last_outer = 0
    for step in range(args.stats_steps):
        epoch = gl.token_count // 60000
        print(gl.token_count)
        if last_outer:
            u.log_scalars({"time/outer": 1000*(time.perf_counter() - last_outer)})
        last_outer = time.perf_counter()

        # compute validation loss
        model.eval()
        if args.swa:
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

        model.skip_forward_hooks = False
        model.skip_backward_hooks = False

        # get gradient values
        with u.timeit("backprop_g"):
            gl.backward_idx = 0
            u.clear_backprops(model)
            output = model(data)
            loss = loss_fn(output, targets)
            loss.backward(retain_graph=True)
        u.log_scalar(loss=loss.item())

        # get Hessian values
        hessian_activations = []
        hessian_backprops = []
        hessians = []   # list of Hessians in Kronecker form

        model.skip_forward_hooks = True
        for (i, layer) in enumerate(model.layers):
            if args.skip_stats:
                continue

            s = AttrDefault(str, {})  # dictionary-like object for layer stats

            #############################
            # Gradient stats
            #############################
            A_t = layer.activations
            assert A_t.shape == (n, d[i])

            # add factor of n because backprop takes loss averaged over batch, while we need per-example loss
            B_t = layer.backprops[0] * n
            assert B_t.shape == (n, d[i + 1])

            G = (B_t, A_t)
            #    g = G.sum(dim=0, keepdim=True) / n  # average gradient
            g = u.kron_sum(G) / n
            assert g.shape == (1, d[i] * d[i + 1])

            s.sparsity = torch.sum(layer.output <= 0) / layer.output.numel()
            s.mean_activation = torch.mean(A_t)
            s.mean_backprop = torch.mean(B_t)

            # empirical Fisher
            with u.timeit(f'sigma-{i}'):
                # efisher = u.kron_cov(G)  # G.t() @ G / n
                sigma = u.kron_sigma(G, g)  #  efisher - g.t() @ g
                s.sigma_l2 = u.kron_sym_l2_norm(sigma)
                s.sigma_erank = u.kron_trace(sigma)/s.sigma_l2  # torch.trace(sigma)/s.sigma_l2

            #############################
            # Hessian stats
            #############################

            # this is a pair of left/right Kronecker fctors
            H = hessians[i]

            with u.timeit(f"invH-{i}"):
                invH = u.kron_inverse(H)

            with u.timeit(f"H_l2-{i}"):
                s.H_l2 = u.kron_sym_l2_norm(H)
                s.iH_l2 = u.kron_sym_l2_norm(invH)

            with u.timeit(f"norms-{i}"):
                s.H_fro = u.kron_fro_norm(H)
                s.invH_fro = u.kron_fro_norm(invH)
                s.grad_fro = u.kron_fro_norm(g)  # g.flatten().norm()
                s.param_fro = layer.weight.data.flatten().norm()

            u.kron_nan_check(H)

            with u.timeit(f"pinvH-{i}"):
                pinvH = u.kron_pinv(H)

            def kron_curv_direction(dd: torch.Tensor):
                """Curvature in direction dd, using factored form"""
                # dd @ H @ dd.t(), computed by kron_quadratic_form(H, dd)
                return u.to_python_scalar(u.kron_quadratic_form(H, dd) / (dd.flatten().norm() ** 2))

            def kron_loss_direction(dd: torch.Tensor, eps):
                """loss improvement if we take step eps in direction dd"""

                # kron_matmul(dd, g) = dd @ g.t()
                return u.to_python_scalar(eps * (u.kron_matmul(dd, g)) - 0.5 * eps ** 2 * u.kron_quadratic_form(H, dd))

            with u.timeit(f'curv-{i}'):
                s.grad_curv = kron_curv_direction(g)
                s.step_openai = 1 / s.grad_curv if s.grad_curv else 999
                s.step_max = 2 / s.H_l2
                s.step_min = torch.tensor(2) / u.kron_trace(H)

                s.regret_gradient = kron_loss_direction(g, s.step_openai)

            with u.timeit(f"batch-{i}"):
                # torch.trace(H @ sigma)                         # (g @ H @ g.t())
                s.batch_openai = u.kron_trace_matmul(H, sigma) / u.kron_quadratic_form(H, g)
                s.diversity = torch.norm(G, "fro") ** 2 / torch.norm(g) ** 2

                # torch.trace(H)
                s.H_erank = u.kron_trace(H) / s.H_l2
                s.batch_jain_simple = 1 + s.H_erank

            u.log_scalars(u.nest_stats(layer.name, s))

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
