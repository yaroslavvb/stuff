"""Train small network on MNIST with Cross-Entropy loss"""

import argparse
import os
import time

import autograd_lib
import globals as gl
# import torch
import torch
import util as u
import wandb
from attrdict import AttrDefault
from torch import nn as nn
from torch.utils.tensorboard import SummaryWriter

# for line profiling
try:
  # noinspection PyUnboundLocalVariable
  profile  # throws an exception when profile isn't defined
except NameError:
  profile = lambda x: x   # if it's not defined simply ignore the decorator.


from train_ciresan import validate


@profile
def main():

    u.install_pdb_handler()
    u.seed_random(1)
    logdir = u.create_local_logdir(args.logdir)
    run_name = os.path.basename(logdir)
    gl.event_writer = SummaryWriter(logdir)
    print(f"Logging to {logdir}")

    loss_type = 'CrossEntropy'

    d1 = args.data_width ** 2
    n = args.stats_batch_size
    o = 10
    d = [d1, o]
    dataset_size = 10000

    model = u.SimpleFullyConnected2(d, bias=True, nonlin=args.nonlin, last_layer_linear=True)
    model = model.to(gl.device)
    print(model)

    try:
        if args.wandb:
            wandb.init(project='curv_train_tiny', name=run_name, dir='/tmp/wandb.runs')
            wandb.tensorboard.patch(tensorboardX=False)
            wandb.config['train_batch'] = args.train_batch_size
            wandb.config['stats_batch'] = args.stats_batch_size
            wandb.config['n'] = n
    except Exception as e:
        print(f"wandb crash with {e}")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
    #  optimizer = torch.optim.Adam(model.parameters(), lr=0.03)  # make 10x smaller for least-squares loss
    dataset = u.TinyMNIST(data_width=args.data_width, dataset_size=dataset_size, loss_type=loss_type)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False, drop_last=True)
    train_iter = u.infinite_iter(train_loader)

    stats_loader = torch.utils.data.DataLoader(dataset, batch_size=args.stats_batch_size, shuffle=False, drop_last=True)
    stats_iter = u.infinite_iter(stats_loader)

    test_dataset = u.TinyMNIST(data_width=args.data_width, train=False, dataset_size=dataset_size, loss_type=loss_type)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.stats_batch_size, shuffle=False, drop_last=True)
    test_iter = u.infinite_iter(test_loader)

    if loss_type == 'LeastSquares':
        loss_fn = u.least_squares
    else:   # loss_type == 'CrossEntropy':
        loss_fn = nn.CrossEntropyLoss()

    autograd_lib.add_hooks(model)
    gl.reset_global_step()
    last_outer = 0
    val_losses = []
    for step in range(args.stats_steps):
        if last_outer:
            u.log_scalars({"time/outer": 1000*(time.perf_counter() - last_outer)})
        last_outer = time.perf_counter()

        with u.timeit("val_loss"):
            test_data, test_targets = next(test_iter)
            test_output = model(test_data)
            val_loss = loss_fn(test_output, test_targets)
            print("val_loss", val_loss.item())
            val_losses.append(val_loss.item())
            u.log_scalar(val_loss=val_loss.item())

        with u.timeit("validate"):
            if loss_type == 'CrossEntropy':
                val_accuracy, val_loss = validate(model, test_loader, f'test (stats_step {step})')
                train_accuracy, train_loss = validate(model, stats_loader, f'train (stats_step {step})')

                metrics = {'stats_step': step, 'val_accuracy': val_accuracy, 'val_loss': val_loss,
                           'train_loss': train_loss, 'train_accuracy': train_accuracy}
                u.log_scalars(metrics)

        data, targets = next(stats_iter)

        if not args.skip_stats:
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
            autograd_lib.disable_hooks()   # TODO(y): use remove_hooks

            with u.timeit("compute_grad1"):
                autograd_lib.compute_grad1(model)
            with u.timeit("compute_hess"):
                autograd_lib.compute_hess(model)

            for (i, layer) in enumerate(model.layers):
                if args.skip_stats:
                    continue

                param_names = {layer.weight: "weight", layer.bias: "bias"}
                for param in [layer.weight, layer.bias]:
                    # input/output layers are unreasonably expensive if not using Kronecker factoring
                    if d[i]*d[i+1] > 2000:
                        print(f'layer {i} is too big ({d[i],d[i+1]}), skipping stats')
                        continue

                    s = AttrDefault(str, {})  # dictionary-like object for layer stats

                    #############################
                    # Gradient stats
                    #############################
                    A_t = layer.activations
                    B_t = layer.backprops_list[0] * n
                    s.sparsity = torch.sum(layer.output <= 0) / layer.output.numel()  # proportion of activations that are zero
                    s.mean_activation = torch.mean(A_t)
                    s.mean_backprop = torch.mean(B_t)

                    # empirical Fisher
                    G = param.grad1.reshape((n, -1))
                    g = G.mean(dim=0, keepdim=True)

                    u.nan_check(G)
                    with u.timeit(f'sigma-{i}'):
                        efisher = G.t() @ G / n
                        sigma = efisher - g.t() @ g
                        # sigma_spectrum =
                        s.sigma_l2 = u.sym_l2_norm(sigma)
                        s.sigma_erank = torch.trace(sigma)/s.sigma_l2

                    H = param.hess
                    lambda_regularizer = args.lmb * torch.eye(H.shape[0]).to(gl.device)
                    u.nan_check(H)

                    with u.timeit(f"invH-{i}"):
                        invH = torch.cholesky_inverse(H+lambda_regularizer)

                    with u.timeit(f"H_l2-{i}"):
                        s.H_l2 = u.sym_l2_norm(H)
                        s.iH_l2 = u.sym_l2_norm(invH)

                    with u.timeit(f"norms-{i}"):
                        s.H_fro = H.flatten().norm()
                        s.iH_fro = invH.flatten().norm()
                        s.grad_fro = g.flatten().norm()
                        s.param_fro = param.data.flatten().norm()

                    def loss_direction(dd: torch.Tensor, eps):
                        """loss improvement if we take step eps in direction dd"""
                        return u.to_python_scalar(eps * (dd @ g.t()) - 0.5 * eps ** 2 * dd @ H @ dd.t())

                    def curv_direction(dd: torch.Tensor):
                        """Curvature in direction dd"""
                        return u.to_python_scalar(dd @ H @ dd.t() / (dd.flatten().norm() ** 2))

                    with u.timeit(f"pinvH-{i}"):
                        pinvH = u.pinv(H)

                    with u.timeit(f'curv-{i}'):
                        s.grad_curv = curv_direction(g)  # curvature (eigenvalue) in direction g
                        ndir = g @ pinvH  # newton direction
                        s.newton_curv = curv_direction(ndir)
                        setattr(layer.weight, 'pre', pinvH)  # save Newton preconditioner
                        s.step_openai = 1 / s.grad_curv if s.grad_curv else 1234567
                        s.step_div_inf = 2 / s.H_l2         # divegent step size for batch_size=infinity
                        s.step_div_1 = torch.tensor(2) / torch.trace(H)   # divergent step for batch_size=1

                        s.newton_fro = ndir.flatten().norm()  # frobenius norm of Newton update
                        s.regret_newton = u.to_python_scalar(g @ pinvH @ g.t() / 2)   # replace with "quadratic_form"
                        s.regret_gradient = loss_direction(g, s.step_openai)

                    with u.timeit(f'rho-{i}'):
                        s.rho, s.lyap_erank, lyap_evals = u.truncated_lyapunov_rho(H, sigma)
                        s.step_div_1_adjusted = s.step_div_1/s.rho

                    with u.timeit(f"batch-{i}"):
                        s.batch_openai = torch.trace(H @ sigma) / (g @ H @ g.t())
                        s.diversity = torch.norm(G, "fro") ** 2 / torch.norm(g) ** 2 / n  # Gradient diversity / n
                        s.noise_variance_pinv = torch.trace(pinvH @ sigma)
                        s.H_erank = torch.trace(H) / s.H_l2
                        s.batch_jain_simple = 1 + s.H_erank
                        s.batch_jain_full = 1 + s.rho * s.H_erank

                    param_name = f"{layer.name}={param_names[param]}"
                    u.log_scalars(u.nest_stats(f"{param_name}", s))

                    H_evals = u.symeig_pos_evals(H)
                    sigma_evals = u.symeig_pos_evals(sigma)
                    u.log_spectrum(f'{param_name}/hess', H_evals)
                    u.log_spectrum(f'{param_name}/sigma', sigma_evals)
                    u.log_spectrum(f'{param_name}/lyap', lyap_evals)

        # gradient steps
        with u.timeit('inner'):
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
                            param.data.mul_(1-args.weight_decay)

                gl.increment_global_step(data.shape[0])

    gl.event_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--wandb', type=int, default=0, help='log to weights and biases')
    parser.add_argument('--autograd_check', type=int, default=0, help='autograd correctness checks')
    parser.add_argument('--logdir', type=str, default='/temp/runs/curv_train_tiny/run')
    parser.add_argument('--train_batch_size', type=int, default=10)
    parser.add_argument('--stats_batch_size', type=int, default=10)
    parser.add_argument('--data_width', type=int, default=8)
    parser.add_argument('--train_steps', type=int, default=10, help="this many train steps between stat collection")
    parser.add_argument('--stats_steps', type=int, default=1000000, help="total number of curvature stats collections")
    parser.add_argument('--nonlin', type=int, default=1, help="whether to add ReLU nonlinearity between layers")
    parser.add_argument('--skip_stats', type=int, default=0, help='skip all stats collection')
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--lmb', type=float, default=1e-3, help="lambda regularizer for inverting H")

    args = parser.parse_args()

    main()
