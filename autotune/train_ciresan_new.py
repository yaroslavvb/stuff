# TODO(y): FRI -- go over all formulas and results from ciresan run
# TODO(y): add angle historgam
# Train Ciresan's 6-layer deep MNIST network.
# (from http://yann.lecun.com/exdb/mnist/)

import argparse
import copy
import os
import sys
import time
import traceback
from collections import defaultdict
from typing import Callable, List

import autograd_lib
import globals as gl
# import torch
import scipy
import torch
import torch.nn as nn
import torchcontrib
import wandb
from attrdict import AttrDefault, AttrDict
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

u.install_pdb_handler()


@profile
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
    parser.add_argument('--hess_samples', type=int, default=1,
                        help='number of samples when sub-sampling outputs, 0 for exact hessian')
    parser.add_argument('--hess_kfac', type=int, default=0, help='whether to use KFAC approximation for hessian')
    parser.add_argument('--compute_rho', type=int, default=0, help='use expensive method to compute rho')
    parser.add_argument('--skip_stats', type=int, default=0, help='skip all stats collection')

    parser.add_argument('--dataset_size', type=int, default=60000)
    parser.add_argument('--train_steps', type=int, default=100, help="this many train steps between stat collection")
    parser.add_argument('--stats_steps', type=int, default=1000000, help="total number of curvature stats collections")

    parser.add_argument('--full_batch', type=int, default=0, help='do stats on the whole dataset')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--dropout', type=int, default=0)
    parser.add_argument('--swa', type=int, default=0)
    parser.add_argument('--lmb', type=float, default=1e-3)

    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--stats_batch_size', type=int, default=10000)
    parser.add_argument('--stats_num_batches', type=int, default=1)
    parser.add_argument('--run_name', type=str, default='noname')
    parser.add_argument('--launch_blocking', type=int, default=0)
    parser.add_argument('--sampled', type=int, default=0)
    parser.add_argument('--curv', type=str, default='kfac',
                        help='decomposition to use for curvature estimates: zero_order, kfac, isserlis or full')
    parser.add_argument('--log_spectra', type=int, default=0)

    u.seed_random(1)
    gl.args = parser.parse_args()
    args = gl.args
    u.seed_random(1)

    gl.project_name = 'train_ciresan'
    u.setup_logdir_and_event_writer(args.run_name)
    print(f"Logging to {gl.logdir}")

    d1 = 28 * 28
    d = [784, 2500, 2000, 1500, 1000, 500, 10]

    # number of samples per datapoint. Used to normalize kfac
    model = u.SimpleFullyConnected2(d, nonlin=args.nonlin, bias=args.bias, dropout=args.dropout)
    model = model.to(gl.device)
    autograd_lib.register(model)

    assert args.dataset_size >= args.stats_batch_size
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    dataset = u.TinyMNIST(data_width=args.data_width, targets_width=args.targets_width, original_targets=True,
                          dataset_size=args.dataset_size)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    train_iter = u.infinite_iter(train_loader)

    assert not args.full_batch, "fixme: validation still uses stats_iter"
    if not args.full_batch:
        stats_loader = torch.utils.data.DataLoader(dataset, batch_size=args.stats_batch_size, shuffle=True,
                                                   drop_last=True)
        stats_iter = u.infinite_iter(stats_loader)
    else:
        stats_iter = None

    test_dataset = u.TinyMNIST(data_width=args.data_width, targets_width=args.targets_width, train=False,
                               original_targets=True,
                               dataset_size=args.dataset_size)
    test_eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.stats_batch_size, shuffle=False,
                                                   drop_last=False)
    train_eval_loader = torch.utils.data.DataLoader(dataset, batch_size=args.stats_batch_size, shuffle=False,
                                                    drop_last=False)

    loss_fn = torch.nn.CrossEntropyLoss()
    autograd_lib.add_hooks(model)
    autograd_lib.disable_hooks()

    gl.token_count = 0
    last_outer = 0

    for step in range(args.stats_steps):
        epoch = gl.token_count // 60000
        lr = optimizer.param_groups[0]['lr']
        print('token_count', gl.token_count)
        if last_outer:
            u.log_scalars({"time/outer": 1000 * (time.perf_counter() - last_outer)})
            print(f'time: {time.perf_counter() - last_outer:.2f}')
        last_outer = time.perf_counter()

        with u.timeit("validate"):
            val_accuracy, val_loss = validate(model, test_eval_loader, f'test (epoch {epoch})')
            train_accuracy, train_loss = validate(model, train_eval_loader, f'train (epoch {epoch})')

        # save log
        metrics = {'epoch': epoch, 'val_accuracy': val_accuracy, 'val_loss': val_loss,
                   'train_loss': train_loss, 'train_accuracy': train_accuracy,
                   'lr': optimizer.param_groups[0]['lr'],
                   'momentum': optimizer.param_groups[0].get('momentum', 0)}
        u.log_scalars(metrics)

        def mom_update(buffer, val):
            buffer *= 0.9
            buffer += val * 0.1

        if not args.skip_stats:
            # number of samples passed through
            n = args.stats_batch_size * args.stats_num_batches

            # quanti
            forward_stats = defaultdict(lambda: AttrDefault(float))

            hessians = defaultdict(lambda: AttrDefault(float))
            jacobians = defaultdict(lambda: AttrDefault(float))
            fishers = defaultdict(lambda: AttrDefault(float))  # empirical fisher/gradient
            quad_fishers = defaultdict(
                lambda: AttrDefault(float))  # gradient statistics that depend on fisher (4th order moments)
            train_regrets = defaultdict(list)
            test_regrets1 = defaultdict(list)
            test_regrets2 = defaultdict(list)
            train_regrets_opt = defaultdict(list)
            test_regrets_opt = defaultdict(list)
            cosines = defaultdict(list)
            dot_products = defaultdict(list)

            current = None

            for i in range(args.stats_num_batches):
                activations = {}
                backprops = {}

                def save_activations(layer, A, _):
                    activations[layer] = A
                    forward_stats[layer].AA += torch.einsum("ni,nj->ij", A, A)

                print('forward')
                with u.timeit("stats_forward"):
                    with autograd_lib.module_hook(save_activations):
                        data, targets = next(stats_iter)
                        output = model(data)
                        loss = loss_fn(output, targets) * len(output)

                def compute_stats(layer, _, B):
                    A = activations[layer]
                    if current == fishers:
                        backprops[layer] = B

                    # about 27ms per layer
                    with u.timeit('compute_stats'):
                        current[layer].BB += torch.einsum("ni,nj->ij", B, B)  # TODO(y): index consistency
                        current[layer].diag += torch.einsum("ni,nj->ij", B * B, A * A)
                        current[layer].BA += torch.einsum("ni,nj->ij", B, A)
                        current[layer].a += torch.einsum("ni->i", A)
                        current[layer].b += torch.einsum("nk->k", B)
                        current[layer].norm2 += ((A * A).sum(dim=1) * (B * B).sum(dim=1)).sum()

                        # compute curvatures in direction of all gradiennts
                        if current is fishers:
                            assert args.stats_num_batches == 1, "not tested on more than one stats step, currently reusing aggregated moments"
                            hess = hessians[layer]
                            jac = jacobians[layer]
                            Bh, Ah = B @ hess.BB / n, A @ forward_stats[layer].AA / n
                            Bj, Aj = B @ jac.BB / n, A @ forward_stats[layer].AA / n
                            norms = ((A * A).sum(dim=1) * (B * B).sum(dim=1))

                            current[layer].min_norm2 = min(norms)
                            current[layer].median_norm2 = torch.median(norms)

                            norms_hess = ((Ah * A).sum(dim=1) * (Bh * B).sum(dim=1))
                            norms_jac = ((Aj * A).sum(dim=1) * (Bj * B).sum(dim=1))

                            current[layer].norm += norms.sum()
                            current[layer].curv_hess += (norms_hess / norms).sum()
                            current[layer].curv_jac += (norms_jac / norms).sum()

                            current[layer].norms_hess += norms_hess.sum()
                            current[layer].norms_jac += norms_jac.sum()

                            normalized_moments = copy.copy(hessians[layer])
                            normalized_moments.AA = forward_stats[layer].AA
                            normalized_moments = u.divide_attributes(normalized_moments, n)

                            train_regrets_ = autograd_lib.offset_losses(A, B, alpha=lr, offset=0, m=normalized_moments,
                                                                        approx=args.curv)
                            test_regrets1_ = autograd_lib.offset_losses(A, B, alpha=lr, offset=1, m=normalized_moments,
                                                                        approx=args.curv)
                            test_regrets2_ = autograd_lib.offset_losses(A, B, alpha=lr, offset=2, m=normalized_moments,
                                                                        approx=args.curv)
                            test_regrets_opt_ = autograd_lib.offset_losses(A, B, alpha=None, offset=2,
                                                                           m=normalized_moments, approx=args.curv)
                            train_regrets_opt_ = autograd_lib.offset_losses(A, B, alpha=None, offset=0,
                                                                            m=normalized_moments, approx=args.curv)
                            cosines_ = autograd_lib.offset_cosines(A, B)
                            train_regrets[layer].extend(train_regrets_)
                            test_regrets1[layer].extend(test_regrets1_)
                            test_regrets2[layer].extend(test_regrets2_)
                            train_regrets_opt[layer].extend(train_regrets_opt_)
                            test_regrets_opt[layer].extend(test_regrets_opt_)
                            cosines[layer].extend(cosines_)
                            dot_products[layer].extend(autograd_lib.offset_dotprod(A, B))

                        # statistics of the form g.Sigma.g
                        elif current == quad_fishers:
                            hess = hessians[layer]
                            sigma = fishers[layer]
                            jac = jacobians[layer]
                            Bs, As = B @ sigma.BB / n, A @ forward_stats[layer].AA / n
                            Bh, Ah = B @ hess.BB / n, A @ forward_stats[layer].AA / n
                            Bj, Aj = B @ jac.BB / n, A @ forward_stats[layer].AA / n

                            norms = ((A * A).sum(dim=1) * (B * B).sum(dim=1))
                            norms_hess = ((Ah * A).sum(dim=1) * (Bh * B).sum(dim=1))
                            norms_jac = ((Aj * A).sum(dim=1) * (Bj * B).sum(dim=1))
                            norms_sigma = ((As * A).sum(dim=1) * (Bs * B).sum(dim=1))

                            current[layer].norm += norms.sum()  # TODO(y) remove, redundant with norm2 above
                            current[layer].curv_sigma += (norms_sigma / norms).sum()
                            current[layer].curv_hess += (norms_hess / norms).sum()
                            current[layer].lyap_hess_sum += (norms_sigma / norms_hess).sum()
                            current[layer].lyap_hess_max = max(norms_sigma/norms_hess)
                            current[layer].lyap_jac_sum += (norms_sigma / norms_jac).sum()
                            current[layer].lyap_jac_max = max(norms_sigma/norms_jac)

                # todo(y): add "compute_fisher" and "compute_jacobian"
                # todo(y): add couple of statistics (effective rank, trace, gradient noise)
                # todo(y): change to stochastic
                # todo(y): plot eigenvalue spectrum for each, OpenAI and Jain stats, gradient noise
                # todo(y): get convolution working
                # todo(y): test on LeNet5
                # todo(y): test on resnet50
                # todo(y): video of spectra changing

                print('backward')
                with u.timeit("backprop_H"):
                    with autograd_lib.module_hook(compute_stats):
                        current = hessians
                        autograd_lib.backward_hessian(output, loss='CrossEntropy', sampled=args.sampled,
                                                      retain_graph=True)  # 600 ms
                        current = jacobians
                        autograd_lib.backward_jacobian(output, sampled=args.sampled, retain_graph=True)  # 600 ms
                        current = fishers
                        model.zero_grad()
                        loss.backward(retain_graph=True)  # 60 ms
                        current = quad_fishers
                        model.zero_grad()
                        loss.backward()  # 60 ms

            print('summarize')
            for (i, layer) in enumerate(model.layers):
                stats_dict = {'hessian': hessians, 'jacobian': jacobians, 'fisher': fishers}

                # evaluate stats from
                # https://app.wandb.ai/yaroslavvb/train_ciresan/runs/425pu650?workspace=user-yaroslavvb
                for stats_name in stats_dict:
                    s = AttrDict()
                    stats = stats_dict[stats_name][layer]

                    for key in forward_stats[layer]:
                        # print(f'copying {key} in {stats_name}, {layer}')
                        try:
                            assert stats[key] == float()
                        except:
                            f"Trying to overwrite {key} in {stats_name}, {layer}"
                        stats[key] = forward_stats[layer][key]

                    diag: torch.Tensor = stats.diag / n

                    # jacobian:
                    # curv in direction of gradient goes down to roughly 0.3-1
                    # maximum curvature goes up to 1000-2000
                    #
                    # Hessian:
                    # max curv goes down to 1, in direction of gradient 0.0001

                    s.diag_l2 = torch.max(diag)  # 40 - 3000 smaller than kfac l2 for jac
                    s.diag_fro = torch.norm(
                        diag)  # jacobian grows to 0.5-1.5, rest falls, layer-5 has phase transition, layer-4 also
                    s.diag_trace = diag.sum()  # jacobian grows 0-1000 (first), 0-150 (last). Almost same as kfac_trace (771 vs 810 kfac). Jacobian has up/down phase transition
                    s.diag_average = diag.mean()
                    # s.diag_erank = s.diag_trace / torch.max(
                    #    diag)  # kind of useless, very large and noise, but layer2/jacobian has up/down phase transition

                    # normalize for mean loss
                    BB = stats.BB / n
                    AA = stats.AA / n
                    # A_evals, _ = torch.symeig(AA)   # averaging 120ms per hit, 90 hits
                    # B_evals, _ = torch.symeig(BB)

                    # s.kfac_l2 = torch.max(A_evals) * torch.max(B_evals)    # 60x larger than diag_l2. layer0/hess has down/up phase transition. layer5/jacobian has up/down phase transition
                    s.kfac_trace = torch.trace(AA) * torch.trace(BB)  # 0/hess down/up tr, 5/jac sharp phase transition
                    s.kfac_fro = torch.norm(stats.AA) * torch.norm(
                        stats.BB)  # 0/hess has down/up tr, 5/jac up/down transition
                    # s.kfac_erank = s.kfac_trace / s.kfac_l2   # first layer has 25, rest 15, all layers go down except last, last noisy
                    # s.kfac_erank_fro = s.kfac_trace / s.kfac_fro / max(stats.BA.shape)

                    s.diversity = (stats.norm2 / n) / u.norm_squared(
                        stats.BA / n)  # gradient diversity. Goes up 3x. Bottom layer has most diversity. Jacobian diversity much less noisy than everythingelse

                    # discrepancy of KFAC based on exact values of diagonal approximation
                    # average difference normalized by average diagonal magnitude
                    diag_kfac = torch.einsum('ll,ii->li', BB, AA)
                    s.kfac_error = (torch.abs(diag_kfac - diag)).mean() / torch.mean(diag.abs())
                    u.log_scalars(u.nest_stats(f'layer-{i}/{stats_name}', s))

                # openai batch size stat
                s = AttrDict()
                hess = hessians[layer]
                jac = jacobians[layer]
                fish = fishers[layer]
                quad_fish = quad_fishers[layer]

                # the following check passes, but is expensive
                # if args.stats_num_batches == 1:
                #    u.check_close(fisher[layer].BA, layer.weight.grad)

                def trsum(A, B):
                    return (A * B).sum()  # computes tr(AB')

                grad = fishers[layer].BA / n
                s.grad_fro = torch.norm(grad)

                # get norms
                hess_A = u.symeig_pos_evals(hess.AA / n)
                hess_B = u.symeig_pos_evals(hess.BB / n)

                jac_A = u.symeig_pos_evals(hess.AA / n)
                jac_B = u.symeig_pos_evals(hess.BB / n)

                s.hess_l2 = max(hess_A) * max(hess_B)
                s.jac_l2 = max(jac_A) * max(jac_B)
                # hess.diag_trace, jac.diag_trace

                s.lyap_hess_max = quad_fish.lyap_hess_max
                s.lyap_hess_ave = quad_fish.lyap_hess_sum / n
                s.lyap_jac_max = quad_fish.lyap_jac_max
                s.lyap_jac_ave = quad_fish.lyap_jac_sum / n
                s.hess_trace = hess.diag.sum() / n
                s.jac_trace = jac.diag.sum() / n

                # Version 1 of Jain stochastic rates, use Hessian for curvature
                b = args.train_batch_size
                s.jain1_sto = s.lyap_hess_max * s.hess_trace / s.lyap_hess_ave
                s.jain1_det = s.hess_l2

                s.jain1_lr = (1/b) * s.jain1_sto + (b-1) / b * s.jain1_det
                s.jain1_lr = 1 / s.jain1_lr

                # Version 2 of Jain stochastic rates, use Jacobian squared for curvature
                s.jain2_sto = s.lyap_jac_max * s.jac_trace / s.lyap_jac_ave
                s.jain2_det = s.jac_l2
                s.jain2_lr = (1/b) * s.jain2_sto + (b-1) / b * s.jain2_det
                s.jain2_lr = 1 / s.jain2_lr

                s.hess_curv = trsum((hess.BB / n) @ grad @ (hess.AA / n), grad) / trsum(grad, grad)
                s.jac_curv = trsum((jac.BB / n) @ grad @ (jac.AA / n), grad) / trsum(grad, grad)

                # compute gradient noise statistics
                # fish.BB has /n factor twice, hence don't need extra /n on fish.AA
                # after sampling, hess_noise,jac_noise became 100x smaller, but normalized is unaffected
                s.hess_noise = (trsum(hess.AA / n, fish.AA / n) * trsum(hess.BB / n, fish.BB / n))
                s.jac_noise = (trsum(jac.AA / n, fish.AA / n) * trsum(jac.BB / n, fish.BB / n))
                s.hess_noise_centered = s.hess_noise - trsum(hess.BB / n @ grad, grad @ hess.AA / n)
                s.jac_noise_centered = s.jac_noise - trsum(jac.BB / n @ grad, grad @ jac.AA / n)

                s.openai_gradient_noise = (fish.norms_hess / n) / trsum(hess.BB / n @ grad, grad @ hess.AA / n)

                s.mean_norm2 = fish.norm2 / n
                s.min_norm2 = fish.min_norm2
                s.median_norm2 = fish.median_norm2
                s.enorms = u.norm_squared(grad)

                s.norms_centered = fish.norm2 / n - u.norm_squared(grad)
                s.norms_hess = fish.norms_hess / n
                s.norms_jac = fish.norms_jac / n

                s.hess_curv_grad = fish.curv_hess / n  # phase transition, hits minimum loss in layer 1, then starts going up. Other layers take longer to reach minimum. Decreases with depth.
                s.sigma_curv_grad = quad_fish.curv_sigma / n
                s.band_bottou = 0.5 * lr * s.sigma_curv_grad / s.hess_curv_grad
                s.band_bottou_stoch = 0.5 * lr * quad_fish.curv_ratio / n
                s.band_yaida = 0.25 * lr * s.mean_norm2
                s.band_yaida_centered = 0.25 * lr * s.norms_centered

                s.jac_curv_grad = fish.curv_jac / n  # this one has much lower variance than jac_curv. Reaches peak at 10k steps, also kfac error reaches peak there. Decreases with depth except for last layer.

                # OpenAI gradient noise statistics
                s.hess_noise_normalized = s.hess_noise_centered / (fish.norms_hess / n)
                s.jac_noise_normalized = s.jac_noise / (fish.norms_jac / n)

                train_regrets_, test_regrets1_, test_regrets2_, train_regrets_opt_, test_regrets_opt_, cosines_, dot_products_ = (torch.stack(r[layer]) for r in (train_regrets, test_regrets1, test_regrets2, train_regrets_opt, test_regrets_opt, cosines, dot_products))
                s.train_regret = train_regrets_.median()  # use median because outliers make it hard to see the trend
                s.test_regret1 = test_regrets1_.median()
                s.test_regret2 = test_regrets2_.median()
                s.test_regret_opt = test_regrets_opt_.median()
                s.train_regret_opt = train_regrets_opt_.median()
                s.mean_dot_product = torch.mean(dot_products_)
                s.median_dot_product = torch.median(dot_products_)
                a = [1, 2, 3]

                s.median_cosine = cosines_.median()
                s.mean_cosine = cosines_.mean()

                # get learning rates
                L1 = s.hess_curv_grad / n
                L2 = s.jac_curv_grad / n
                diversity = (fish.norm2 / n) / u.norm_squared(grad)
                robust_diversity = (fish.norm2 / n) / fish.median_norm2
                dotprod_diversity = fish.median_norm2 / s.median_dot_product
                s.lr1 = 2 / (L1 * diversity)
                s.lr2 = 2 / (L2 * diversity)
                s.lr3 = 2 / (L2 * robust_diversity)

                s.lr4 = 2 / (L2 * dotprod_diversity)

                s.regret_ratio = (
                            train_regrets_opt_ / test_regrets_opt_).median()  # ratio between train and test regret, large means overfitting
                u.log_scalars(u.nest_stats(f'layer-{i}', s))

                def erank(vals): return vals.sum() / vals.max()
                def srank(vals): return (vals * vals).sum() / (vals.max() ** 2)

                # compute stats that would let you bound rho
                # if i == 0:
                #     hhh = hessians[model.layers[-1]].BB
                #     fff = fishers[model.layers[-1]].BB
                #     d = fff.shape[0]
                #     L = u.lyapunov_spectral(hhh, 2 * fff, cond=1e-5)
                #     mismatch = torch.eig(fff @ u.pinv(hhh, cond=1e-5))[0]
                #     mismatch = mismatch[:, 0]  # extract real part
                #     mismatch = mismatch.sort()[0]
                #     mismatch = torch.flip(mismatch, [0])
                #
                #     u.log_scalars({f'layer-{i}/rho': d/erank(u.symeig_pos_evals(L))})
                #     u.log_scalars({f'layer-{i}/rho_cheap': d/erank(mismatch)})
                #     u.log_spectrum(f'layer-{i}/sigma', u.symeig_pos_evals(fff), loglog=False)
                #     u.log_spectrum(f'layer-{i}/hess', u.symeig_pos_evals(hhh), loglog=False)
                #     u.log_spectrum(f'layer-{i}/lyapunov', u.symeig_pos_evals(L), loglog=False)
                #     u.log_spectrum(f'layer-{i}/lyapunov_cheap', mismatch, loglog=False)

                if args.log_spectra:
                    with u.timeit('spectrum'):
                        hess_A = u.symeig_pos_evals(hess.AA / n)
                        u.log_spectrum(f'layer-{i}/hess_A', hess_A)
                        hess_B = u.symeig_pos_evals(hess.BB / n)
                        u.log_spectrum(f'layer-{i}/hess_B', hess_B)

                        fish_A = u.symeig_pos_evals(fish.AA / n)
                        u.log_spectrum(f'layer-{i}/fish_A', hess_A)
                        fish_B = u.symeig_pos_evals(fish.BB / n)
                        u.log_spectrum(f'layer-{i}/fish_B', hess_B)

                        u.log_scalars({f'layer-{i}/trace_ratio': fish_B.sum()/hess_B.sum()})

                        # hess_evals = u.outer(hess_A, hess_B).flatten()

                        u.log_scalars({f'layer-{i}/hessA_erank': erank(hess_A)})
                        u.log_scalars({f'layer-{i}/hessB_erank': erank(hess_B)})
                        u.log_scalars({f'layer-{i}/fishA_erank': erank(fish_A)})
                        u.log_scalars({f'layer-{i}/fishB_erank': erank(fish_B)})

                        L = torch.eig(u.lyapunov_spectral(hess.BB, 2*fish.BB, cond=1e-8))[0]
                        L = L[:, 0]  # extract real part
                        L = L.sort()[0]
                        L = torch.flip(L, [0])

                        L_cheap = torch.eig(fish.BB @ u.pinv(hess.BB, cond=1e-8))[0]
                        L_cheap = L_cheap[:, 0]  # extract real part
                        L_cheap = L_cheap.sort()[0]
                        L_cheap = torch.flip(L_cheap, [0])

                        d = len(hess_B)

                        u.log_spectrum(f'layer-{i}/Lyap', L)
                        u.log_spectrum(f'layer-{i}/Lyap_cheap', L_cheap)

                        u.log_scalars({f'layer-{i}/dims': d})
                        u.log_scalars({f'layer-{i}/L_erank': erank(L)})
                        u.log_scalars({f'layer-{i}/L_cheap_erank': erank(L_cheap)})

                        u.log_scalars({f'layer-{i}/rho': d/erank(L)})
                        u.log_scalars({f'layer-{i}/rho_cheap': d/erank(L_cheap)})

                # 1. x norms histogram (jacobian norms)
                # 2. gradient norms histogram
                # 3.
                # step size stat
                # rho?

                # todo(y): add weight magnitude
                # todo(y): add curvatures in direction of mean gradient
                # todo(y): add regret
                # todo(y): log spectra
                # todo(y): gradient norms histogram
                # TODO(y): check mean error again, check hess_noise, jac_noise

        model.train()
        with u.timeit('train'):
            for i in range(args.train_steps):
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
                            param.data.mul_(1 - args.weight_decay)

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
    print(
        f'Eval: Average {tag} loss: {val_loss:.4f}, Accuracy: {correct:.0f}/{len(val_loader.dataset)} ({val_accuracy:.2f}%)')

    return val_accuracy, val_loss


if __name__ == '__main__':
    main()
