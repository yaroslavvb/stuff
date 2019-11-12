# Take simple MNIST model, test that line-search in Newton direction finds optimum
# Additionally test Hessian manual vs autograd computation.
# This functio

import argparse
import json
import os
import random
import shutil
import sys

import numpy as np
import torch.nn as nn
from attrdict import AttrDefault
from torch import optim

import util as u

import globals as gl


try:
    import wandb
except Exception as e:
    pass
    # print(f"wandb crash with {e}")

from torchvision import datasets, transforms
import torch

from torchcurv.optim import SecondOrderOptimizer
from torchcurv.utils import Logger

DATASET_MNIST = 'MNIST'
IMAGE_SIZE = 28
NUM_CHANNELS = 1

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def install_pdb_handler():
    """Automatically start pdb:
      1. CTRL+\\ breaks into pdb.
      2. pdb gets launched on exception.
  """

    import signal
    import pdb

    def handler(_signum, _frame):
        pdb.set_trace()

    signal.signal(signal.SIGQUIT, handler)

    # Drop into PDB on exception
    # from https://stackoverflow.com/questions/13174412
    def info(type_, value, tb):
        if hasattr(sys, 'ps1') or not sys.stderr.isatty():
            # we are in interactive mode or we don't have a tty-like
            # device, so we call the default hook
            sys.__excepthook__(type_, value, tb)
        else:
            import traceback
            import pdb
            # we are NOT in interactive mode, print the exception...
            traceback.print_exception(type_, value, tb)
            print()
            # ...then start the debugger in post-mortem mode.
            pdb.pm()

    sys.excepthook = info


install_pdb_handler()


class FastMNIST(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Normalize it with the usual MNIST mean and std
        # self.data = self.data.sub_(0.1307).div_(0.3081)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


class FastBinaryMNIST(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Normalize it with the usual MNIST mean and std
        # self.data = self.data.sub_(0.1307).div_(0.3081)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data.to(device), self.targets.to(device)
        self.targets = (self.targets > 0).float()
        self.targets.unsqueeze(1)
        assert len(self.targets.shape) == 2

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


class SimpleMNIST(datasets.MNIST):
    """Simple dataset where goal is to predict sum of pixels."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Normalize it with the usual MNIST mean and std
        self.data = self.data.sub_(0.1307).div_(0.3081)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data.to(device), self.targets.to(device)
        self.target = self.data.sum(dim=(2, 3)).squeeze(1)
        self.targets = self.targets.unsqueeze(1)
        assert len(self.targets.shape) == 2

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


def compute_loss(output, target):
    err = output - target.float()
    return torch.sum(err * err) / 2 / output.shape[0]


logger = None


def log(metrics, step):
    global logger
    # print(metrics)
    try:
        logger.write(metrics)
    except:  # crashes with JSON conversion error sometimes
        pass

    try:
        wandb.log(metrics, step=step)
    except:
        pass


def test_lineasearch():
    """Implement linesearch with sanity checks."""
    global logger, stats_data, stats_targets, args
    run_name = 'default'  # name of run in

    torch.set_default_dtype(torch.float32)

    # Copy this file & config to args.out
    out = '/tmp'
    if not os.path.isdir(out):
        os.makedirs(out)
    shutil.copy(os.path.realpath(__file__), out)

    # Setup logger
    log_file_name = 'log'
    logger = Logger(out, log_file_name)
    logger.start()

    # Set device
    use_cuda = False
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Set random seed
    u.seed_random(1)

    # Setup data augmentation & data pre processing
    train_transforms, val_transforms = [], []

    train_transforms.append(transforms.ToTensor())
    val_transforms.append(transforms.ToTensor())

    train_transform = transforms.Compose(train_transforms)
    # val_transform = transforms.Compose(val_transforms)

    num_classes = 10
    dataset_class = SimpleMNIST

    class Net(nn.Module):
        def __init__(self, d, nonlin=True):
            super().__init__()
            self.layers = []
            self.all_layers = []
            self.d = d
            for i in range(len(d) - 1):
                linear = nn.Linear(d[i], d[i + 1], bias=False)
                self.layers.append(linear)
                self.all_layers.append(linear)
                if nonlin:
                    self.all_layers.append(nn.ReLU())
            self.predict = torch.nn.Sequential(*self.all_layers)

        def forward(self, x: torch.Tensor):
            x = x.reshape((-1, self.d[0]))
            return self.predict(x)

    stats_batch_size = 1
    def compute_layer_stats(layer):
        stats = AttrDefault(str, {})
        n = stats_batch_size
        param = u.get_param(layer)
        d = len(param.flatten())
        layer_idx = model.layers.index(layer)
        assert layer_idx >= 0
        assert stats_data.shape[0] == n

        def backprop_loss():
            model.zero_grad()
            output = model(stats_data)  # use last saved data batch for backprop
            loss = compute_loss(output, stats_targets)
            loss.backward()
            return loss, output

        def backprop_output():
            model.zero_grad()
            output = model(stats_data)
            output.backward(gradient=torch.ones_like(output))
            return output

        # per-example gradients, n, d
        loss, output = backprop_loss()
        At = layer.data_input
        Bt = layer.grad_output * n
        G = u.khatri_rao_t(At, Bt)
        g = G.sum(dim=0, keepdim=True) / n
        u.check_close(g, u.vec(param.grad).t())

        stats.diversity = torch.norm(G, "fro") ** 2 / g.flatten().norm() ** 2

        stats.gradient_norm = g.flatten().norm()
        stats.parameter_norm = param.data.flatten().norm()
        pos_activations = torch.sum(layer.data_output > 0)
        neg_activations = torch.sum(layer.data_output <= 0)
        stats.sparsity = pos_activations.float()/(pos_activations+neg_activations)

        output = backprop_output()
        At2 = layer.data_input
        u.check_close(At, At2)
        B2t = layer.grad_output
        J = u.khatri_rao_t(At, B2t)
        H = J.t() @ J / n

        model.zero_grad()
        output = model(stats_data)  # use last saved data batch for backprop
        loss = compute_loss(output, stats_targets)
        hess = u.hessian(loss, param)

        hess = hess.transpose(2, 3).transpose(0, 1).reshape(d, d)
        u.check_close(hess, H)
        u.check_close(hess, H)

        stats.hessian_norm = u.l2_norm(H)
        stats.jacobian_norm = u.l2_norm(J)
        Joutput = J.sum(dim=0) / n
        stats.jacobian_sensitivity = Joutput.norm()

        # newton decrement
        stats.loss_newton = u.to_python_scalar(g @ u.pinv(H) @ g.t() / 2)
        u.check_close(stats.loss_newton, loss)

        # do line-search to find optimal step
        def line_search(directionv, start, end, steps=10):
            """Takes steps between start and end, returns steps+1 loss entries"""
            param0 = param.data.clone()
            param0v = u.vec(param0).t()
            losses = []
            for i in range(steps+1):
                output = model(stats_data)  # use last saved data batch for backprop
                loss = compute_loss(output, stats_targets)
                losses.append(loss)
                offset = start+i*((end-start)/steps)
                param1v = param0v + offset*directionv

                param1 = u.unvec(param1v.t(), param.data.shape[0])
                param.data.copy_(param1)

            output = model(stats_data)  # use last saved data batch for backprop
            loss = compute_loss(output, stats_targets)
            losses.append(loss)

            param.data.copy_(param0)
            return losses

        # try to take a newton step
        gradv = g
        line_losses = line_search(-gradv @ u.pinv(H), 0, 2, steps=10)
        u.check_equal(line_losses[0], loss)
        u.check_equal(line_losses[6], 0)
        assert line_losses[5] > line_losses[6]
        assert line_losses[7] > line_losses[6]
        return stats

    train_dataset = dataset_class(
        root='/tmp/data', train=True, download=True, transform=train_transform)

    batch_size = 32
    num_workers = 0
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    stats_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=stats_batch_size, shuffle=False, num_workers=num_workers)
    stats_data, stats_targets = next(iter(stats_loader))

    model = Net([NUM_CHANNELS * IMAGE_SIZE ** 2, 8, 8, 1],  nonlin=False)
    setattr(model, 'num_classes', num_classes)
    model = model.to(device)
    u.freeze(model.layers[0])
    u.freeze(model.layers[2])

    # use learning rate 0 to avoid changing parameter vector
    optim_kwargs = dict(lr=0.001, momentum=0.9, weight_decay=0, l2_reg=0,
                        bias_correction=False, acc_steps=1,
                        curv_type="Cov", curv_shapes={"Linear": "Kron"},
                        momentum_type="preconditioned", update_inv=False, precondition_grad=False)
    curv_args = dict(damping=0, ema_decay=1)
    SecondOrderOptimizer(model, **optim_kwargs, curv_kwargs=curv_args)  # call optimizer to add backward hoooks
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    start_epoch = 1

    # Run training
    epochs = 100
    for epoch in range(start_epoch, epochs + 1):
        num_examples_processed = epoch * len(train_loader) * train_loader.batch_size
        layer_stats = compute_layer_stats(model.layers[1])


def train(model, device, train_loader, optimizer, epoch, args, logger):
    global global_data, global_target
    model.train()

    loss = None
    confidence = {'top1': 0, 'top1_true': 0, 'top1_false': 0, 'true': 0, 'false': 0}
    total_data_size = 0
    epoch_size = len(train_loader.dataset)
    num_iters_in_epoch = len(train_loader)
    base_num_iter = (epoch - 1) * num_iters_in_epoch

    last_elapsed_time = 0
    interval_ms = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        global_data = data
        global_target = target

        for name, param in model.named_parameters():
            attr = 'p_pre_{}'.format(name)
            setattr(model, attr, param.detach().clone())

        # update params
        def closure():
            optimizer.zero_grad()
            output = model(data)
            loss = compute_loss(output, target)
            loss.backward(create_graph=False)
            return loss, output

        #loss, output = optimizer.step(closure=closure)
        loss, output = closure()
        optimizer.step()
        loss = loss.item()

        iteration = base_num_iter + batch_idx + 1
        total_data_size += len(data)

        batch_size = 128
        if batch_idx % 20 == 0:
            elapsed_time = logger.elapsed_time
            if last_elapsed_time:
                interval_ms = 1000 * (elapsed_time - last_elapsed_time)
            last_elapsed_time = elapsed_time
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, '
                  'Accuracy: {:.0f}/{} ({:.2f}%), '
                  'Elapsed Time: {:.1f}s'.format(epoch, total_data_size, epoch_size, 100. * (batch_idx + 1) / num_iters_in_epoch, loss, 0, total_data_size, 0, elapsed_time))

            # save log
            lr = optimizer.param_groups[0]['lr']
            metrics = {'epoch': epoch, 'iteration': iteration, 'elapsed_time': elapsed_time,
                       'accuracy': 0, 'loss': loss, 'lr': lr, 'step_ms': interval_ms}

            for name, param in model.named_parameters():
                attr = 'p_pre_{}'.format(name)
                p_pre = getattr(model, attr)
                p_norm = param.norm().item()
                p_shape = list(param.size())
                p_pre_norm = p_pre.norm().item()
                if param.grad is not None:
                    g_norm = param.grad.norm().item()
                else:
                    g_norm = 0
                upd_norm = param.sub(p_pre).norm().item()
                noise_scale = getattr(param, 'noise_scale', 0)

                p_log = {'p_shape': p_shape, 'p_norm': p_norm, 'p_pre_norm': p_pre_norm,
                         'g_norm': g_norm, 'upd_norm': upd_norm, 'noise_scale': noise_scale}
                #  print(p_log)
                metrics[name] = p_log

            log(metrics, step=iteration * batch_size)

    return 0, loss, confidence


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--dataset', type=str,
                        choices=[DATASET_MNIST], default=DATASET_MNIST,
                        help='name of dataset')
    parser.add_argument('--root', type=str, default='./data',
                        help='root of dataset')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--stats_batch_size', type=int, default=1,
                        help='size of batch to use for second order statistics')
    parser.add_argument('--val_batch_size', type=int, default=128,
                        help='input batch size for valing')
    parser.add_argument('--normalizing_data', action='store_true',
                        help='[data pre processing] normalizing data')
    parser.add_argument('--random_crop', action='store_true',
                        help='[data augmentation] random crop')
    parser.add_argument('--random_horizontal_flip', action='store_true',
                        help='[data augmentation] random horizontal flip')
    # Training Settings
    parser.add_argument('--arch_file', type=str, default=None,
                        help='name of file which defines the architecture')
    parser.add_argument('--arch_name', type=str, default='LeNet5',
                        help='name of the architecture')
    parser.add_argument('--arch_args', type=json.loads, default=None,
                        help='[JSON] arguments for the architecture')
    parser.add_argument('--optim_name', type=str, default=SecondOrderOptimizer.__name__,
                        help='name of the optimizer')
    parser.add_argument('--optim_args', type=json.loads, default=None,
                        help='[JSON] arguments for the optimizer')
    parser.add_argument('--curv_args', type=json.loads, default=None,
                        help='[JSON] arguments for the curvature')
    # Options
    parser.add_argument('--download', action='store_true', default=True,
                        help='if True, downloads the dataset (CIFAR-10 or 100) from the internet')
    parser.add_argument('--create_graph', action='store_true', default=False,
                        help='create graph of the derivative')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of sub processes for data loading')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log_file_name', type=str, default='log',
                        help='log file name')
    parser.add_argument('--checkpoint_interval', type=int, default=50,
                        help='how many epochs to wait before logging training status')
    parser.add_argument('--resume', type=str, default=None,
                        help='checkpoint path for resume training')
    parser.add_argument('--out', type=str, default='/tmp',
                        help='dir to save output files')
    parser.add_argument('--config', default=None,
                        help='config file path')
    parser.add_argument('--fisher_mc_approx', action='store_true', default=False,
                        help='if True, Fisher is estimated by MC sampling')
    parser.add_argument('--fisher_num_mc', type=int, default=1,
                        help='number of MC samples for estimating Fisher')
    parser.add_argument('--log_wandb', type=int, default=0,
                        help='log to wandb')

    args = parser.parse_args()

    u.run_all_tests(sys.modules[__name__])
