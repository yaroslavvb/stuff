# To verify Newton convergence in 1 step
# python train_tiny.py --wandb=0 --method=newton --nonlin=0 --layer=0

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
import wandb
from attrdict import AttrDefault
from torch.utils.tensorboard import SummaryWriter

import util as u


def main():
    attemp_count = 0
    while os.path.exists(f"{args.logdir}{attemp_count:02d}"):
        attemp_count += 1
    logdir = f"{args.logdir}{attemp_count:02d}"

    run_name = os.path.basename(logdir)
    gl.event_writer = SummaryWriter(logdir)
    print(f"Logging to {run_name}")
    u.seed_random(1)

    d1 = args.data_width ** 2
    d2 = 10
    d3 = args.targets_width ** 2
    o = d3
    n = args.stats_batch_size
    d = [d1, d2, d3]
    model = u.SimpleFullyConnected(d, nonlin=args.nonlin)
    model = model.to(gl.device)

    try:
        # os.environ['WANDB_SILENT'] = 'true'
        if args.wandb:
            wandb.init(project='curv_train_tiny', name=run_name)
            wandb.tensorboard.patch(tensorboardX=False)
            wandb.config['train_batch'] = args.train_batch_size
            wandb.config['stats_batch'] = args.stats_batch_size
            wandb.config['method'] = args.method
            wandb.config['d1'] = d1
            wandb.config['d2'] = d2
            wandb.config['d3'] = d3
            wandb.config['n'] = n
    except Exception as e:
        print(f"wandb crash with {e}")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)

    dataset = u.TinyMNIST(data_width=args.data_width, targets_width=args.targets_width,
                          dataset_size=args.dataset_size)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False, drop_last=True)
    train_iter = u.infinite_iter(train_loader)

    stats_loader = torch.utils.data.DataLoader(dataset, batch_size=args.stats_batch_size, shuffle=False, drop_last=True)
    stats_iter = u.infinite_iter(stats_loader)

    test_dataset = u.TinyMNIST(data_width=args.data_width, targets_width=args.targets_width,
                               dataset_size=args.dataset_size, train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.stats_batch_size, shuffle=True, drop_last=True)
    test_iter = u.infinite_iter(test_loader)

    skip_forward_hooks = False
    skip_backward_hooks = False

    def capture_activations(module: nn.Module, input: List[torch.Tensor], output: torch.Tensor):
        if skip_forward_hooks:
            return
        assert not hasattr(module, 'activations'), "Seeing results of previous autograd, call util.zero_grad to clear"
        assert len(input) == 1, "this was tested for single input layers only"
        setattr(module, "activations", input[0].detach())
        setattr(module, "output", output.detach())

    def capture_backprops(module: nn.Module, _input, output):
        if skip_backward_hooks:
            return
        assert len(output) == 1, "this works for single variable layers only"
        if gl.backward_idx == 0:
            assert not hasattr(module, 'backprops'), "Seeing results of previous autograd, call util.zero_grad to clear"
            setattr(module, 'backprops', [])
        assert gl.backward_idx == len(module.backprops)
        module.backprops.append(output[0])

    def save_grad(param: nn.Parameter) -> Callable[[torch.Tensor], None]:
        """Hook to save gradient into 'param.saved_grad', so it can be accessed after model.zero_grad(). Only stores gradient
        if the value has not been set, call util.zero_grad to clear it."""

        def save_grad_fn(grad):
            if not hasattr(param, 'saved_grad'):
                setattr(param, 'saved_grad', grad)

        return save_grad_fn

    for layer in model.layers:
        layer.register_forward_hook(capture_activations)
        layer.register_backward_hook(capture_backprops)
        layer.weight.register_hook(save_grad(layer.weight))

    def loss_fn(data, targets):
        err = data - targets.view(-1, data.shape[1])
        assert len(data) == len(targets)
        return torch.sum(err * err) / 2 / len(data)

    gl.token_count = 0
    last_outer = 0
    for step in range(args.stats_steps):
        if last_outer:
            u.log_scalars({"time/outer": 1000*(time.perf_counter() - last_outer)})
        last_outer = time.perf_counter()
        # compute validation loss
        skip_forward_hooks = True
        skip_backward_hooks = True
        with u.timeit("val_loss"):
            test_data, test_targets = next(test_iter)
            test_output = model(test_data)
            val_loss = loss_fn(test_output, test_targets)
            print("val_loss", val_loss.item())
            u.log_scalar(val_loss=val_loss.item())

        # compute stats
        data, targets = next(stats_iter)
        skip_forward_hooks = False
        skip_backward_hooks = False

        # get gradient values
        with u.timeit("backprop_g"):
            gl.backward_idx = 0
            u.zero_grad(model)
            output = model(data)
            loss = loss_fn(output, targets)
            loss.backward(retain_graph=True)

        # get Hessian values
        skip_forward_hooks = True
        id_mat = torch.eye(o).to(gl.device)

        u.log_scalar(loss=loss.item())

        with u.timeit("backprop_H"):
            # optionally use randomized low-rank approximation of Hessian
            hess_rank = args.hess_samples if args.hess_samples else o

            for out_idx in range(hess_rank):
                model.zero_grad()
                # backprop to get section of batch output jacobian for output at position out_idx
                output = model(data)  # opt: using autograd.grad means I don't have to zero_grad
                if args.hess_samples:
                    bval = torch.LongTensor(n, o).to(gl.device).random_(0, 2) * 2 - 1
                    bval = bval.float()
                else:
                    ei = id_mat[out_idx]
                    bval = torch.stack([ei] * n)
                gl.backward_idx = out_idx + 1
                output.backward(bval)
            skip_backward_hooks = True  #

        for (i, layer) in enumerate(model.layers):
            s = AttrDefault(str, {})  # dictionary-like object for layer stats

            #############################
            # Gradient stats
            #############################
            A_t = layer.activations
            assert A_t.shape == (n, d[i])

            # add factor of n because backprop takes loss averaged over batch, while we need per-example loss
            B_t = layer.backprops[0] * n
            assert B_t.shape == (n, d[i + 1])

            with u.timeit(f"khatri_g-{i}"):
                G = u.khatri_rao_t(B_t, A_t)  # batch loss Jacobian
            assert G.shape == (n, d[i] * d[i + 1])
            g = G.sum(dim=0, keepdim=True) / n  # average gradient
            assert g.shape == (1, d[i] * d[i + 1])

            if args.autograd_check:
                u.check_close(B_t.t() @ A_t / n, layer.weight.saved_grad)
                u.check_close(g.reshape(d[i + 1], d[i]), layer.weight.saved_grad)

            s.sparsity = torch.sum(layer.output <= 0) / layer.output.numel()
            s.mean_activation = torch.mean(A_t)
            s.mean_backprop = torch.mean(B_t)

            # empirical Fisher
            with u.timeit(f'sigma-{i}'):
                efisher = G.t() @ G / n
                sigma = efisher - g.t() @ g
                s.sigma_l2 = u.sym_l2_norm(sigma)
                s.sigma_erank = torch.trace(sigma)/s.sigma_l2

            #############################
            # Hessian stats
            #############################
            A_t = layer.activations
            Bh_t = [layer.backprops[out_idx + 1] for out_idx in range(hess_rank)]
            Amat_t = torch.cat([A_t] * hess_rank, dim=0)
            Bmat_t = torch.cat(Bh_t, dim=0)

            assert Amat_t.shape == (n * hess_rank, d[i])
            assert Bmat_t.shape == (n * hess_rank, d[i + 1])

            lambda_regularizer = args.lmb * torch.eye(d[i] * d[i + 1]).to(gl.device)
            with u.timeit(f"khatri_H-{i}"):
                Jb = u.khatri_rao_t(Bmat_t, Amat_t)  # batch Jacobian, in row-vec format

            with u.timeit(f"H-{i}"):
                H = Jb.t() @ Jb / n

            with u.timeit(f"invH-{i}"):
                invH = torch.cholesky_inverse(H+lambda_regularizer)

            with u.timeit(f"H_l2-{i}"):
                s.H_l2 = u.sym_l2_norm(H)
                s.iH_l2 = u.sym_l2_norm(invH)

            with u.timeit(f"norms-{i}"):
                s.H_fro = H.flatten().norm()
                s.iH_fro = invH.flatten().norm()
                s.jacobian_fro = Jb.flatten().norm()
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

            with u.timeit("pinvH"):
                pinvH = u.pinv(H)

            with u.timeit(f'curv-{i}'):
                s.regret_newton = u.to_python_scalar(g @ pinvH @ g.t() / 2)
                s.grad_curv = curv_direction(g)
                ndir = g @ pinvH  # newton direction
                s.newton_curv = curv_direction(ndir)
                setattr(layer.weight, 'pre', pinvH)  # save Newton preconditioner
                s.step_openai = 1 / s.grad_curv if s.grad_curv else 999
                s.step_max = 2 / u.sym_l2_norm(H)
                s.step_min = torch.tensor(2) / torch.trace(H)

                s.newton_fro = ndir.flatten().norm()  # frobenius norm of Newton update
                s.regret_gradient = loss_direction(g, s.step_openai)

            with u.timeit(f'rho-{i}'):
                p_sigma = u.lyapunov_svd(H, sigma)
                if u.has_nan(p_sigma) and args.compute_rho:  # use expensive method
                    H0 = H.cpu().detach().numpy()
                    sigma0 = sigma.cpu().detach().numpy()
                    p_sigma = scipy.linalg.solve_lyapunov(H0, sigma0)
                    p_sigma = torch.tensor(p_sigma).to(gl.device)

                if u.has_nan(p_sigma):
                    s.psigma_erank = H.shape[0]
                    s.rho = 1
                else:
                    s.psigma_erank = u.sym_erank(p_sigma)
                    s.rho = H.shape[0] / s.psigma_erank

            with u.timeit(f"batch-{i}"):
                s.batch_openai = torch.trace(H @ sigma) / (g @ H @ g.t())
                print('openai batch', s.batch_openai)
                s.diversity = torch.norm(G, "fro") ** 2 / torch.norm(g) ** 2

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

            u.log_scalar(train_loss=loss.item())

            if args.method != 'newton':
                optimizer.step()
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
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

    parser.add_argument('--train_batch_size', type=int, default=3)
    parser.add_argument('--stats_batch_size', type=int, default=1000)
    parser.add_argument('--dataset_size', type=int, default=10000)
    parser.add_argument('--train_steps', type=int, default=10, help="this many train steps between stat collection")
    parser.add_argument('--stats_steps', type=int, default=1000, help="total number of curvature stats collections")
    parser.add_argument('--nonlin', type=int, default=1, help="whether to add ReLU nonlinearity between layers")
    parser.add_argument('--method', type=str, choices=['gradient', 'newton'], default='gradient',
                        help="descent method, newton or gradient")
    parser.add_argument('--layer', type=int, default=-1, help="restrict updates to this layer")
    parser.add_argument('--data_width', type=int, default=8)
    parser.add_argument('--targets_width', type=int, default=4)
    parser.add_argument('--lmb', type=float, default=1e-3)
    parser.add_argument('--hess_samples', type=int, default=1, help='number of samples when sub-sampling outputs, 0 for exact hessian')
    parser.add_argument('--hess_kfac', type=int, default=0, help='whether to use KFAC approximation for hessian')
    parser.add_argument('--compute_rho', type=int, default=0, help='use expensive method to compute rho')

    args = parser.parse_args()

    main()
