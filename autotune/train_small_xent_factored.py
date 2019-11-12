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
    args.stats_batch_size = min(args.stats_batch_size, args.dataset_size)
    args.train_batch_size = min(args.train_batch_size, args.dataset_size)
    n = args.stats_batch_size
    o = 10
    d = [d1, 60, 60, 60, o]
    # dataset_size = args.dataset_size

    model = u.SimpleFullyConnected2(d, bias=True, nonlin=args.nonlin, last_layer_linear=True)
    model = model.to(gl.device)
    u.mark_expensive(model.layers[0])    # to stop grad1/hess calculations on this layer
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

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    #  optimizer = torch.optim.Adam(model.parameters(), lr=0.03)  # make 10x smaller for least-squares loss
    dataset = u.TinyMNIST(data_width=args.data_width, dataset_size=args.dataset_size, loss_type=loss_type)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False, drop_last=True)
    train_iter = u.infinite_iter(train_loader)

    stats_loader = torch.utils.data.DataLoader(dataset, batch_size=args.stats_batch_size, shuffle=False, drop_last=True)
    stats_iter = u.infinite_iter(stats_loader)
    stats_data, stats_targets = next(stats_iter)

    test_dataset = u.TinyMNIST(data_width=args.data_width, train=False, dataset_size=args.dataset_size, loss_type=loss_type)
    test_batch_size = min(args.dataset_size, 1000)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, drop_last=True)
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
                # train_accuracy, train_loss = validate(model, train_loader, f'train (stats_step {step})')

                metrics = {'stats_step': step, 'val_accuracy': val_accuracy, 'val_loss': val_loss}
                u.log_scalars(metrics)

        data, targets = stats_data, stats_targets

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
    parser.add_argument('--lr', type=float, default=0.03, help='learning rate')
    parser.add_argument('--autograd_check', type=int, default=0, help='autograd correctness checks')
    parser.add_argument('--logdir', type=str, default='/temp/runs/curv_train_tiny/run')
    parser.add_argument('--train_batch_size', type=int, default=10)
    parser.add_argument('--stats_batch_size', type=int, default=10)
    parser.add_argument('--dataset_size', type=int, default=500)
    parser.add_argument('--data_width', type=int, default=8)
    parser.add_argument('--train_steps', type=int, default=10, help="this many train steps between stat collection")
    parser.add_argument('--stats_steps', type=int, default=1000000, help="total number of curvature stats collections")
    parser.add_argument('--nonlin', type=int, default=1, help="whether to add ReLU nonlinearity between layers")
    parser.add_argument('--skip_stats', type=int, default=0, help='skip all stats collection')
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--lmb', type=float, default=1e-3, help="lambda regularizer for inverting H")

    args = parser.parse_args()

    main()
