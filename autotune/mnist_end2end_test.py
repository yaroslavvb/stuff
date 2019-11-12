import argparse
import os
import time

import autograd_lib
import globals as gl
# import torch
import scipy
import torch
import util as u
import wandb
from attrdict import AttrDefault, AttrDict
from torch.utils.tensorboard import SummaryWriter

from torch import nn as nn

# for line profiling
try:
    # noinspection PyUnboundLocalVariable
    profile  # throws an exception when profile isn't defined
except NameError:
    profile = lambda x: x  # if it's not defined simply ignore the decorator.


def test_main():

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
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
    parser.add_argument('--logdir', type=str, default='/temp/runs/curv_train_tiny/run')

    parser.add_argument('--train_batch_size', type=int, default=100)
    parser.add_argument('--stats_batch_size', type=int, default=60000)
    parser.add_argument('--dataset_size', type=int, default=60000)
    parser.add_argument('--train_steps', type=int, default=100, help="this many train steps between stat collection")
    parser.add_argument('--stats_steps', type=int, default=1000000, help="total number of curvature stats collections")
    parser.add_argument('--nonlin', type=int, default=1, help="whether to add ReLU nonlinearity between layers")
    parser.add_argument('--method', type=str, choices=['gradient', 'newton'], default='gradient',
                        help="descent method, newton or gradient")
    parser.add_argument('--layer', type=int, default=-1, help="restrict updates to this layer")
    parser.add_argument('--data_width', type=int, default=28)
    parser.add_argument('--targets_width', type=int, default=28)
    parser.add_argument('--lmb', type=float, default=1e-3)
    parser.add_argument('--hess_samples', type=int, default=1,
                        help='number of samples when sub-sampling outputs, 0 for exact hessian')
    parser.add_argument('--hess_kfac', type=int, default=0, help='whether to use KFAC approximation for hessian')
    parser.add_argument('--compute_rho', type=int, default=1, help='use expensive method to compute rho')
    parser.add_argument('--skip_stats', type=int, default=0, help='skip all stats collection')
    parser.add_argument('--full_batch', type=int, default=0, help='do stats on the whole dataset')
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    #args = parser.parse_args()
    args = AttrDict()
    args.lmb = 1e-3
    args.compute_rho = 1
    args.weight_decay = 1e-4
    args.method = 'gradient'
    args.logdir = '/tmp'
    args.data_width = 2
    args.targets_width = 2
    args.train_batch_size = 10
    args.full_batch = False
    args.skip_stats = False
    args.autograd_check = False


    u.seed_random(1)
    logdir = u.create_local_logdir(args.logdir)
    run_name = os.path.basename(logdir)
    #gl.event_writer = SummaryWriter(logdir)
    gl.event_writer = u.NoOp()
    # print(f"Logging to {run_name}")


    # small values for debugging
    # loss_type = 'LeastSquares'
    loss_type = 'CrossEntropy'

    args.wandb = 0
    args.stats_steps = 10
    args.train_steps = 10
    args.stats_batch_size = 10
    args.data_width = 2
    args.targets_width = 2
    args.nonlin = False
    d1 = args.data_width ** 2
    d2 = 2
    d3 = args.targets_width ** 2

    d1 = args.data_width ** 2
    assert args.data_width == args.targets_width
    o = d1
    n = args.stats_batch_size
    d = [d1, 30, 30, 30, 20, 30, 30, 30, d1]

    if loss_type == 'CrossEntropy':
        d3 = 10
    o = d3
    n = args.stats_batch_size
    d = [d1, d2, d3]
    dsize = max(args.train_batch_size, args.stats_batch_size) + 1

    model = u.SimpleFullyConnected2(d, bias=True, nonlin=args.nonlin)
    model = model.to(gl.device)

    try:
        # os.environ['WANDB_SILENT'] = 'true'
        if args.wandb:
            wandb.init(project='curv_train_tiny', name=run_name)
            wandb.tensorboard.patch(tensorboardX=False)
            wandb.config['train_batch'] = args.train_batch_size
            wandb.config['stats_batch'] = args.stats_batch_size
            wandb.config['method'] = args.method
            wandb.config['n'] = n
    except Exception as e:
        print(f"wandb crash with {e}")

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)  # make 10x smaller for least-squares loss
    dataset = u.TinyMNIST(data_width=args.data_width, targets_width=args.targets_width, dataset_size=dsize,
                          original_targets=True)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False, drop_last=True)
    train_iter = u.infinite_iter(train_loader)

    stats_iter = None
    if not args.full_batch:
        stats_loader = torch.utils.data.DataLoader(dataset, batch_size=args.stats_batch_size, shuffle=False,
                                                   drop_last=True)
        stats_iter = u.infinite_iter(stats_loader)

    test_dataset = u.TinyMNIST(data_width=args.data_width, targets_width=args.targets_width, train=False,
                               dataset_size=dsize, original_targets=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.train_batch_size, shuffle=False,
                                              drop_last=True)
    test_iter = u.infinite_iter(test_loader)

    if loss_type == 'LeastSquares':
        loss_fn = u.least_squares
    elif loss_type == 'CrossEntropy':
        loss_fn = nn.CrossEntropyLoss()

    autograd_lib.add_hooks(model)
    gl.token_count = 0
    last_outer = 0
    val_losses = []
    for step in range(args.stats_steps):
        if last_outer:
            u.log_scalars({"time/outer": 1000 * (time.perf_counter() - last_outer)})
        last_outer = time.perf_counter()

        with u.timeit("val_loss"):
            test_data, test_targets = next(test_iter)
            test_output = model(test_data)
            val_loss = loss_fn(test_output, test_targets)
            # print("val_loss", val_loss.item())
            val_losses.append(val_loss.item())
            u.log_scalar(val_loss=val_loss.item())

        # compute stats
        if args.full_batch:
            data, targets = dataset.data, dataset.targets
        else:
            data, targets = next(stats_iter)

        # Capture Hessian and gradient stats
        autograd_lib.enable_hooks()
        autograd_lib.clear_backprops(model)
        autograd_lib.clear_hess_backprops(model)
        with u.timeit("backprop_g"):
            output = model(data)
            loss = loss_fn(output, targets)
            loss.backward(retain_graph=True)
        with u.timeit("backprop_H"):
            autograd_lib.backprop_hess(output, hess_type=loss_type)
        autograd_lib.disable_hooks()  # TODO(y): use remove_hooks

        with u.timeit("compute_grad1"):
            autograd_lib.compute_grad1(model)
        with u.timeit("compute_hess"):
            autograd_lib.compute_hess(model)

        for (i, layer) in enumerate(model.layers):

            # input/output layers are unreasonably expensive if not using Kronecker factoring
            if d[i] > 50 or d[i + 1] > 50:
                print(f'layer {i} is too big ({d[i], d[i + 1]}), skipping stats')
                continue

            if args.skip_stats:
                continue

            s = AttrDefault(str, {})  # dictionary-like object for layer stats

            #############################
            # Gradient stats
            #############################
            A_t = layer.activations
            assert A_t.shape == (n, d[i])

            # add factor of n because backprop takes loss averaged over batch, while we need per-example loss
            B_t = layer.backprops_list[0] * n
            assert B_t.shape == (n, d[i + 1])

            with u.timeit(f"khatri_g-{i}"):
                G = u.khatri_rao_t(B_t, A_t)  # batch loss Jacobian
            assert G.shape == (n, d[i] * d[i + 1])
            g = G.sum(dim=0, keepdim=True) / n  # average gradient
            assert g.shape == (1, d[i] * d[i + 1])

            u.check_equal(G.reshape(layer.weight.grad1.shape), layer.weight.grad1)

            if args.autograd_check:
                u.check_close(B_t.t() @ A_t / n, layer.weight.saved_grad)
                u.check_close(g.reshape(d[i + 1], d[i]), layer.weight.saved_grad)

            s.sparsity = torch.sum(layer.output <= 0) / layer.output.numel()  # proportion of activations that are zero
            s.mean_activation = torch.mean(A_t)
            s.mean_backprop = torch.mean(B_t)

            # empirical Fisher
            with u.timeit(f'sigma-{i}'):
                efisher = G.t() @ G / n
                sigma = efisher - g.t() @ g
                s.sigma_l2 = u.sym_l2_norm(sigma)
                s.sigma_erank = torch.trace(sigma) / s.sigma_l2

            lambda_regularizer = args.lmb * torch.eye(d[i + 1] * d[i]).to(gl.device)
            H = layer.weight.hess

            with u.timeit(f"invH-{i}"):
                invH = torch.cholesky_inverse(H + lambda_regularizer)

            with u.timeit(f"H_l2-{i}"):
                s.H_l2 = u.sym_l2_norm(H)
                s.iH_l2 = u.sym_l2_norm(invH)

            with u.timeit(f"norms-{i}"):
                s.H_fro = H.flatten().norm()
                s.iH_fro = invH.flatten().norm()
                s.grad_fro = g.flatten().norm()
                s.param_fro = layer.weight.data.flatten().norm()

            u.nan_check(H)
            if args.autograd_check:
                model.zero_grad()
                output = model(data)
                loss = loss_fn(output, targets)
                H_autograd = u.hessian(loss, layer.weight)
                H_autograd = H_autograd.reshape(d[i] * d[i + 1], d[i] * d[i + 1])
                u.check_close(H, H_autograd)

            #  u.dump(sigma, f'/tmp/sigmas/H-{step}-{i}')
            def loss_direction(dd: torch.Tensor, eps):
                """loss improvement if we take step eps in direction dd"""
                return u.to_python_scalar(eps * (dd @ g.t()) - 0.5 * eps ** 2 * dd @ H @ dd.t())

            def curv_direction(dd: torch.Tensor):
                """Curvature in direction dd"""
                return u.to_python_scalar(dd @ H @ dd.t() / (dd.flatten().norm() ** 2))

            with u.timeit(f"pinvH-{i}"):
                pinvH = H.pinverse()

            with u.timeit(f'curv-{i}'):
                s.grad_curv = curv_direction(g)
                ndir = g @ pinvH  # newton direction
                s.newton_curv = curv_direction(ndir)
                setattr(layer.weight, 'pre', pinvH)  # save Newton preconditioner
                s.step_openai = s.grad_fro ** 2 / s.grad_curv if s.grad_curv else 999
                s.step_max = 2 / s.H_l2
                s.step_min = torch.tensor(2) / torch.trace(H)

                s.newton_fro = ndir.flatten().norm()  # frobenius norm of Newton update
                s.regret_newton = u.to_python_scalar(g @ pinvH @ g.t() / 2)  # replace with "quadratic_form"
                s.regret_gradient = loss_direction(g, s.step_openai)

            with u.timeit(f'rho-{i}'):
                p_sigma = u.lyapunov_spectral(H, sigma)

                discrepancy = torch.max(abs(p_sigma - p_sigma.t()) / p_sigma)

                s.psigma_erank = u.sym_erank(p_sigma)
                s.rho = H.shape[0] / s.psigma_erank

            with u.timeit(f"batch-{i}"):
                s.batch_openai = torch.trace(H @ sigma) / (g @ H @ g.t())
                s.diversity = torch.norm(G, "fro") ** 2 / torch.norm(g) ** 2 / n

                # Faster approaches for noise variance computation
                # s.noise_variance = torch.trace(H.inverse() @ sigma)
                # try:
                #     # this fails with singular sigma
                #     s.noise_variance = torch.trace(torch.solve(sigma, H)[0])
                #     # s.noise_variance = torch.trace(torch.lstsq(sigma, H)[0])
                #     pass
                # except RuntimeError as _:
                s.noise_variance_pinv = torch.trace(pinvH @ sigma)

                s.H_erank = torch.trace(H) / s.H_l2
                s.batch_jain_simple = 1 + s.H_erank
                s.batch_jain_full = 1 + s.rho * s.H_erank

            u.log_scalars(u.nest_stats(layer.name, s))

        # gradient steps
        with u.timeit('inner'):
            for i in range(args.train_steps):
                optimizer.zero_grad()
                data, targets = next(train_iter)
                model.zero_grad()
                output = model(data)
                loss = loss_fn(output, targets)
                loss.backward()

                #            u.log_scalar(train_loss=loss.item())

                if args.method != 'newton':
                    optimizer.step()
                    if args.weight_decay:
                        for group in optimizer.param_groups:
                            for param in group['params']:
                                param.data.mul_(1 - args.weight_decay)
                else:
                    for (layer_idx, layer) in enumerate(model.layers):
                        param: torch.nn.Parameter = layer.weight
                        param_data: torch.Tensor = param.data
                        param_data.copy_(param_data - 0.1 * param.grad)
                        if layer_idx != 1:  # only update 1 layer with Newton, unstable otherwise
                            continue
                        u.nan_check(layer.weight.pre)
                        u.nan_check(param.grad.flatten())
                        u.nan_check(u.v2r(param.grad.flatten()) @ layer.weight.pre)
                        param_new_flat = u.v2r(param_data.flatten()) - u.v2r(param.grad.flatten()) @ layer.weight.pre
                        u.nan_check(param_new_flat)
                        param_data.copy_(param_new_flat.reshape(param_data.shape))

                gl.token_count += data.shape[0]

    gl.event_writer.close()

    assert val_losses[0] > 2.4  # 2.4828238487243652
    assert val_losses[-1] < 2.25  # 2.20609712600708


if __name__ == '__main__':

    test_main()
