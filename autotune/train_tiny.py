# To verify Newton convergence in 1 step
# python train_tiny.py --wandb=0 --method=newton --nonlin=0 --layer=0

import argparse
import os
import sys
from typing import Callable

import globals as gl
# import torch
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

    try:
        # os.environ['WANDB_SILENT'] = 'true'
        if args.wandb:
            wandb.init(project='curv_train_tiny', name=run_name)
            wandb.tensorboard.patch(tensorboardX=False)
            wandb.config['train_batch'] = args.train_batch_size
            wandb.config['stats_batch'] = args.stats_batch_size
            wandb.config['method'] = args.method

    except Exception as e:
        print(f"wandb crash with {e}")

    #    data_width = 4
    #    targets_width = 2

    d1 = args.data_width ** 2
    d2 = 10
    d3 = args.targets_width ** 2
    o = d3
    n = args.stats_batch_size
    d = [d1, d2, d3]
    model = u.SimpleFullyConnected(d, nonlin=args.nonlin)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    dataset = u.TinyMNIST(data_width=args.data_width, targets_width=args.targets_width,
                          dataset_size=args.dataset_size)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False, drop_last=True)
    train_iter = u.infinite_iter(train_loader)

    stats_loader = torch.utils.data.DataLoader(dataset, batch_size=args.stats_batch_size, shuffle=False, drop_last=True)
    stats_iter = u.infinite_iter(stats_loader)

    def capture_activations(module, input, _output):
        if skip_forward_hooks:
            return
        assert gl.backward_idx == 0   # no need to forward-prop on Hessian computation
        assert not hasattr(module, 'activations'), "Seeing activations from previous forward, call util.zero_grad to clear"
        assert len(input) == 1, "this works for single input layers only"
        setattr(module, "activations", input[0].detach())

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
    for step in range(args.stats_steps):
        data, targets = next(stats_iter)
        skip_forward_hooks = False
        skip_backward_hooks = False

        # get gradient values
        gl.backward_idx = 0
        u.zero_grad(model)
        output = model(data)
        loss = loss_fn(output, targets)
        loss.backward(retain_graph=True)

        print("loss", loss.item())

        # get Hessian values
        skip_forward_hooks = True
        id_mat = torch.eye(o)

        u.log_scalars({'loss': loss.item()})

        # o = 0
        for out_idx in range(o):
            model.zero_grad()
            # backprop to get section of batch output jacobian for output at position out_idx
            output = model(data)  # opt: using autograd.grad means I don't have to zero_grad
            ei = id_mat[out_idx]
            bval = torch.stack([ei] * n)
            gl.backward_idx = out_idx+1
            output.backward(bval)
        skip_backward_hooks = True  #

        for (i, layer) in enumerate(model.layers):
            s = AttrDefault(str, {})   # dictionary-like object for layer stats

            #############################
            # Gradient stats
            #############################
            A_t = layer.activations
            assert A_t.shape == (n, d[i])

            # add factor of n because backprop takes loss averaged over batch, while we need per-example loss
            B_t = layer.backprops[0] * n
            assert B_t.shape == (n, d[i+1])

            G = u.khatri_rao_t(B_t, A_t)           # batch loss Jacobian
            assert G.shape == (n, d[i]*d[i+1])
            g = G.sum(dim=0, keepdim=True) / n     # average gradient
            assert g.shape == (1, d[i]*d[i+1])

            if args.autograd_check:
                u.check_close(B_t.t() @ A_t / n, layer.weight.saved_grad)
                u.check_close(g.reshape(d[i+1], d[i]), layer.weight.saved_grad)

            # empirical Fisher
            efisher = G.t() @ G / n
            sigma = efisher - g.t() @ g
            # u.dump(sigma, f'/tmp/sigmas/{step}-{i}')
            s.sigma_l2 = u.l2_norm(sigma)

            #############################
            # Hessian stats
            #############################
            A_t = layer.activations
            Bh_t = [layer.backprops[out_idx+1] for out_idx in range(o)]
            Amat_t = torch.cat([A_t] * o, dim=0)
            Bmat_t = torch.cat(Bh_t, dim=0)

            assert Amat_t.shape == (n*o, d[i])
            assert Bmat_t.shape == (n*o, d[i+1])

            Jb = u.khatri_rao_t(Bmat_t, Amat_t)   # batch Jacobian, in row-vec format
            H = Jb.t() @ Jb / n
            pinvH = u.pinv(H)

            s.hess_l2 = u.l2_norm(H)
            s.invhess_l2 = u.l2_norm(pinvH)

            s.hess_fro = H.flatten().norm()
            s.invhess_fro = pinvH.flatten().norm()

            s.jacobian_l2 = u.l2_norm(Jb)
            s.grad_fro = g.flatten().norm()
            s.param_fro = layer.weight.data.flatten().norm()

            u.nan_check(H)
            if args.autograd_check:
                model.zero_grad()
                output = model(data)
                loss = loss_fn(output, targets)
                H_autograd = u.hessian(loss, layer.weight)
                H_autograd = H_autograd.reshape(d[i] * d[i+1], d[i] * d[i+1])
                u.check_close(H, H_autograd)

            #  u.dump(sigma, f'/tmp/sigmas/H-{step}-{i}')
            def loss_direction(dd: torch.Tensor, eps):
                """loss improvement if we take step eps in direction dd"""
                return u.to_python_scalar(eps * (dd @ g.t()) - 0.5 * eps ** 2 * dd @ H @ dd.t())

            def curv_direction(dd: torch.Tensor):
                """Curvature in direction dd"""
                return u.to_python_scalar(dd @ H @ dd.t() / dd.flatten().norm() ** 2)

            s.regret_newton = u.to_python_scalar(g @ u.pinv(H) @ g.t() / 2)
            s.grad_curv = curv_direction(g)
            ndir = g @ u.pinv(H)   # newton direction
            s.newton_curv = curv_direction(ndir)
            setattr(layer.weight, 'pre', u.pinv(H))       # save Newton preconditioner
            s.step_openai = 1/s.grad_curv if s.grad_curv else 999

            s.newton_fro = ndir.flatten().norm()   # frobenius norm of Newton update
            s.regret_gradient = loss_direction(g, s.step_openai)

            u.log_scalars(u.nest_stats(layer.name, s))

        # gradient steps
        for i in range(args.train_steps):
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
                    param_data.copy_(param_data - 0.1*param.grad)
                    if layer_idx != 1:   # only update 1 layer with Newton, unstable otherwise
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
    parser.add_argument('--autograd_check', type=int, default=1, help='autograd correctness checks')
    parser.add_argument('--logdir', type=str, default='/temp/runs/curv_train_tiny/run')

    parser.add_argument('--train_batch_size', type=int, default=3)
    parser.add_argument('--stats_batch_size', type=int, default=100)
    parser.add_argument('--dataset_size', type=int, default=100)
    parser.add_argument('--train_steps', type=int, default=300, help="this many train steps between stat collection")
    parser.add_argument('--stats_steps', type=int, default=10, help="total number of curvature stats collections")
    parser.add_argument('--nonlin', type=int, default=1, help="whether to add ReLU nonlinearity between layers")
    parser.add_argument('--method', type=str, choices=['gradient', 'newton'], default='gradient',
                        help="descent method, newton or gradient")
    parser.add_argument('--layer', type=int, default=-1, help="restrict updates to this layer")
    parser.add_argument('--data_width', type=int, default=4)
    parser.add_argument('--targets_width', type=int, default=2)

    args = parser.parse_args()

    main()
