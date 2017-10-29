import util as u

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

# todo: make images global

step = 0
final_loss = None

def benchmark(batch_size, iters, seed=1, cuda=True, history=100, verbose=False):
  global step, final_loss
  
  step = 0
  final_loss = None

  torch.manual_seed(seed)
  np.random.seed(seed)
  if cuda:
    torch.cuda.manual_seed(seed)

  visible_size = 28*28
  hidden_size = 196
  
  images = torch.Tensor(u.get_mnist_images(batch_size).T)
  images = images[:batch_size]
  if cuda:
    images = images.cuda()
  data = Variable(images)

  class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.encoder = nn.Parameter(torch.rand(visible_size, hidden_size))

    def forward(self, input):
      x = input.view(-1, visible_size)
      x = torch.sigmoid(torch.mm(x, self.encoder))
      x = torch.sigmoid(torch.mm(x, torch.transpose(self.encoder, 0, 1)))
      return x.view_as(input)

  # initialize model and weights
  model = Net()
  model.encoder.data = torch.Tensor(u.ng_init(visible_size,
                                              hidden_size))
  if cuda:
    model.cuda()
  
  model.train()
  optimizer = optim.LBFGS(model.parameters(), max_iter=iters, history_size=history, lr=1.0)

  times = []
  def closure():
    global step, final_loss
    optimizer.zero_grad()
    output = model(data)
    loss = F.mse_loss(output, data)
    if verbose:
      loss0 = loss.data[0]
      times.append(u.last_time())
      print("Step %3d loss %6.5f msec %6.3f"%(step, loss0, u.last_time()))
    step+=1
    if step == iters:
      final_loss = loss.data[0]
    loss.backward()
    u.record_time()
    return loss
  
  optimizer.step(closure)

  output = model(data)
  loss = F.mse_loss(output, data)
  loss0 = loss.data[0]

  if verbose:
    u.summarize_time()

    #  print(times)
  s = ','.join(["%f"%(n,) for n in times[2:]])
  print('{', s,'}')
  
  return final_loss



def main():
  import common_gd
  args = common_gd.args
  args.cuda = not args.no_cuda and torch.cuda.is_available()

  print(benchmark(batch_size=args.batch_size, iters=args.iters, seed=args.seed, cuda = args.cuda, history=args.history, verbose=True))

if __name__=='__main__':
  main()
