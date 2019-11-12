# Plot simple minimization problem in wandb

import argparse
import json
import os
import random
import shutil
import sys
from typing import Any, Dict

import globals as g
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from attrdict import AttrDefault
#  from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from torch import optim


#def log_tb(tag, val):
#    """Log value to tensorboard (relies on g.token_count rather than step count to give comparable graphs across
#    batch sizes)"""
#    g.event_writer.add_scalar(tag, val, g.token_count)


import util as u


try:
    import wandb
except Exception as e:
    pass
    # print(f"wandb crash with {e}")

from torchvision import datasets, transforms
import torch

from torchcurv.optim import SecondOrderOptimizer, VIOptimizer
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
        self.targets = self.data.sum(dim=(2, 3))
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


def log_scalars(metrics: Dict[str, Any], parent_tag: str = '') -> None:
    for tag in metrics:
        g.event_writer.add_scalar(tag=tag, scalar_value=metrics[tag], global_step=g.token_count)
    #    g.event_writer.add_scalars(main_tag=tag, tag_scalar_dict=metrics, get_global_step=g.token_count)
    #    try:
    #        wandb.log(metrics, step=step)
    #    except:
    #        pass


def main():
    global logger, stats_data, stats_targets

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
    parser.add_argument('--logdir', type=str, default='/temp/graph_test/run',
                        help='dir to save output files')
    parser.add_argument('--out', type=str, default=None,
                        help='dir to save output files')
    parser.add_argument('--config', default=None,
                        help='config file path')
    parser.add_argument('--fisher_mc_approx', action='store_true', default=False,
                        help='if True, Fisher is estimated by MC sampling')
    parser.add_argument('--fisher_num_mc', type=int, default=1,
                        help='number of MC samples for estimating Fisher')
    parser.add_argument('--log_wandb', type=int, default=1,
                        help='log to wandb')

    args = parser.parse_args()

    # get run name from logdir
    root_logdir=args.logdir
    count = 0
    while os.path.exists(f"{root_logdir}{count:02d}"):
        count += 1
    args.logdir = f"{root_logdir}{count:02d}"

    run_name = os.path.basename(args.logdir)

    assert args.out is None, "Use args.logdir instead of args.out"
    args.out = args.logdir

    # Copy this file & config to args.out
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    shutil.copy(os.path.realpath(__file__), args.out)

    # Load config file
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        dict_args = vars(args)
        dict_args.update(config)

    if args.config is not None:
        shutil.copy(args.config, args.out)
    if args.arch_file is not None:
        shutil.copy(args.arch_file, args.out)

    # Setup logger
    logger = Logger(args.out, args.log_file_name)
    logger.start()

    g.event_writer = SummaryWriter(args.logdir)

    try:
        # os.environ['WANDB_SILENT'] = 'true'
        if args.log_wandb:
            wandb.init(project='test-graphs_test', name=run_name)
            wandb.tensorboard.patch(tensorboardX=False)
        wandb.config['config'] = args.config
        wandb.config['batch'] = args.batch_size
        wandb.config['optim'] = args.optim_name
    except Exception as e:
        if args.log_wandb:
            print(f"wandb crash with {e}")
        pass

    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup data augmentation & data pre processing
    train_transforms, val_transforms = [], []
    if args.random_crop:
        train_transforms.append(transforms.RandomCrop(32, padding=4))

    if args.random_horizontal_flip:
        train_transforms.append(transforms.RandomHorizontalFlip())

    train_transforms.append(transforms.ToTensor())
    val_transforms.append(transforms.ToTensor())

    if args.normalizing_data:
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_transforms.append(normalize)
        val_transforms.append(normalize)

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

    def compute_layer_stats(layer):
        refreeze = False
        if hasattr(layer, 'frozen') and layer.frozen:
            u.unfreeze(layer)
            refreeze = True

        s = AttrDefault(str, {})
        n = args.stats_batch_size
        param = u.get_param(layer)
        _d = len(param.flatten())   # dimensionality of parameters
        layer_idx = model.layers.index(layer)
        # TODO: get layer type, include it in name
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
        _loss, _output = backprop_loss()
        At = layer.data_input
        Bt = layer.grad_output * n
        G = u.khatri_rao_t(At, Bt)
        g = G.sum(dim=0, keepdim=True) / n
        u.check_close(g, u.vec(param.grad).t())

        s.diversity = torch.norm(G, "fro") ** 2 / g.flatten().norm() ** 2
        s.grad_fro = g.flatten().norm()
        s.param_fro = param.data.flatten().norm()
        pos_activations = torch.sum(layer.data_output > 0)
        neg_activations = torch.sum(layer.data_output <= 0)
        s.a_sparsity = neg_activations.float()/(pos_activations+neg_activations)  # 1 sparsity means all 0's
        activation_size = len(layer.data_output.flatten())
        s.a_magnitude = torch.sum(layer.data_output)/activation_size

        _output = backprop_output()
        B2t = layer.grad_output
        J = u.khatri_rao_t(At, B2t)  # batch output Jacobian
        H = J.t() @ J / n

        s.hessian_l2 = u.l2_norm(H)
        s.jacobian_l2 = u.l2_norm(J)
        J1 = J.sum(dim=0) / n        # single output Jacobian
        s.J1_l2 = J1.norm()

        # newton decrement
        def loss_direction(direction, eps):
            """loss improvement if we take step eps in direction dir"""
            return u.to_python_scalar(eps * (direction @ g.t()) - 0.5 * eps ** 2 * direction @ H @ direction.t())

        s.regret_newton = u.to_python_scalar(g @ u.pinv(H) @ g.t() / 2)

        # TODO: gradient diversity is stuck at 1
        # TODO: newton/gradient angle
        # TODO: newton step magnitude
        s.grad_curvature = u.to_python_scalar(g @ H @ g.t())  # curvature in direction of g
        s.step_openai = u.to_python_scalar(s.grad_fro ** 2 / s.grad_curvature) if s.grad_curvature else 999

        s.regret_gradient = loss_direction(g, s.step_openai)

        if refreeze:
            u.freeze(layer)
        return s

    train_dataset = dataset_class(
        root=gl.dataset, train=True, download=args.download, transform=train_transform)
    # val_dataset = t(root=args.root, train=False, download=args.download, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers)
    stats_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.stats_batch_size, shuffle=False, num_workers=args.num_workers)
    stats_data, stats_targets = next(iter(stats_loader))

    arch_kwargs = {} if args.arch_args is None else args.arch_args
    arch_kwargs['num_classes'] = num_classes

    model = Net([NUM_CHANNELS * IMAGE_SIZE ** 2, 8, 1],  nonlin=False)
    setattr(model, 'num_classes', num_classes)
    model = model.to(device)
    param = u.get_param(model.layers[0])
    param.data.copy_(torch.ones_like(param)/len(param.data.flatten()))
    param = u.get_param(model.layers[1])
    param.data.copy_(torch.ones_like(param)/len(param.data.flatten()))

    u.freeze(model.layers[0])

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
    for epoch in range(start_epoch, args.epochs + 1):
        num_examples_processed = epoch * len(train_loader) * train_loader.batch_size
        g.token_count = num_examples_processed
        for i in range(len(model.layers)):
            if i == 0:
                continue    # skip initial expensive layer

            layer = model.layers[i]
            layer_stats = compute_layer_stats(layer)
            layer_name = f"{i:02d}-{layer.__class__.__name__.lower()}"
            log_scalars(u.nest_stats(f'stats/{layer_name}', layer_stats))

        # train
        accuracy, loss, confidence = train(model, device, train_loader, optimizer, epoch, args, logger)

        # save log
        iteration = epoch * len(train_loader)
        metrics = {'epoch': epoch, 'iteration': iteration,
                   'accuracy': accuracy, 'loss': loss,
                   'val_accuracy': 0, 'val_loss': 0,
                   'lr': optimizer.param_groups[0]['lr'],
                   'momentum': optimizer.param_groups[0].get('momentum', 0)}
        log_scalars(metrics)

        # save checkpoint
        if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
            path = os.path.join(args.out, 'epoch{}.ckpt'.format(epoch))
            data = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(data, path)


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
            loss.backward(create_graph=args.create_graph)
            return loss, output

        loss, output = closure()
        optimizer.step()
        loss = loss.item()

        iteration = base_num_iter + batch_idx + 1
        total_data_size += len(data)

        if batch_idx % args.log_interval == 0:
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

            log_scalars(metrics)
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

                p_log = {'p_shape': p_shape[0], 'p_norm': p_norm, 'p_pre_norm': p_pre_norm,
                         'g_norm': g_norm, 'upd_norm': upd_norm, 'noise_scale': noise_scale}
                #  print(p_log)
                metrics[name] = p_log
                log_scalars(u.nest_stats(f'kazuki/{name}', p_log))

    return 0, loss, confidence


def validate(model, device, val_loader, optimizer):
    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in val_loader:

            data, target = data.to(device), target.to(device)

            if isinstance(optimizer, VIOptimizer):
                output = optimizer.prediction(data)
            else:
                output = model(data)

            val_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = 100. * correct / len(val_loader.dataset)

    print('\nEval: Average loss: {:.4f}, Accuracy: {:.0f}/{} ({:.2f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset), val_accuracy))

    return val_accuracy, val_loss


if __name__ == '__main__':
    main()
