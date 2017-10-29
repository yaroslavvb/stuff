import argparse
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

parser.add_argument('--batch-size', type=int, default=60000, metavar='N',
                    help='input batch size for training')
parser.add_argument('--iters', type=int, default=100, metavar='N',
                    help='number of iterations to run for (default: 20)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--hidden-size', type=int, default=196, metavar='H',
                    help='hidden size')
parser.add_argument('--visible-size', type=int, default=784, metavar='V',
                    help='visible-size')
parser.add_argument('--gd', action='store_true', default=False,
                    help='force run of gradient descent instead of lbfgs')
parser.add_argument('--history', type=int, default=100, metavar='V',
                    help='history buffer for lbfgs')
args = parser.parse_args()
