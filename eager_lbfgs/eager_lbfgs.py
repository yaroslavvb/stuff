import util as u

import tensorflow as tf
import numpy as np
import time

from tensorflow.contrib.eager.python import tfe
tfe.enable_eager_execution()

def dot(a, b):
  """Dot product function since TensorFlow doesn't have one."""
  return tf.reduce_sum(a*b)

def verbose_func(s):
  print(s)

final_loss = None
times = []
def lbfgs(opfunc, x, config, state, do_verbose):
  """port of lbfgs.lua, using TensorFlow eager mode.
  """

  global final_loss, times
  
  maxIter = config.maxIter or 20
  maxEval = config.maxEval or maxIter*1.25
  tolFun = config.tolFun or 1e-5
  tolX = config.tolX or 1e-9
  nCorrection = config.nCorrection or 100
  lineSearch = config.lineSearch
  lineSearchOpts = config.lineSearchOptions
  learningRate = config.learningRate or 1
  isverbose = config.verbose or False

  # verbose function
  if isverbose:
    verbose = verbose_func
  else:
    verbose = lambda x: None

    # evaluate initial f(x) and df/dx
  f, g = opfunc(x)

  f_hist = [f]
  currentFuncEval = 1
  state.funcEval = state.funcEval + 1
  p = g.shape[0]

  # check optimality of initial point
  tmp1 = tf.abs(g)
  if tf.reduce_sum(tmp1) <= tolFun:
    verbose("optimality condition below tolFun")
    return x, f_hist

  # optimize for a max of maxIter iterations
  nIter = 0
  times = []
  while nIter < maxIter:
    start_time = time.time()
    
    # keep track of nb of iterations
    nIter = nIter + 1
    state.nIter = state.nIter + 1

    ############################################################
    ## compute gradient descent direction
    ############################################################
    if state.nIter == 1:
      d = -g
      old_dirs = []
      old_stps = []
      Hdiag = 1
    else:
      # do lbfgs update (update memory)
      y = g - g_old
      s = d*t
      ys = dot(y, s)
      
      if ys > 1e-10:
        # updating memory
        if len(old_dirs) == nCorrection:
          # shift history by one (limited-memory)
          del old_dirs[0]
          del old_stps[0]

        # store new direction/step
        old_dirs.append(s)
        old_stps.append(y)

        # update scale of initial Hessian approximation
        Hdiag = ys/dot(y, y)

      # compute the approximate (L-BFGS) inverse Hessian 
      # multiplied by the gradient
      k = len(old_dirs)

      # need to be accessed element-by-element, so don't re-type tensor:
      ro = [0]*nCorrection
      for i in range(k):
        ro[i] = 1/dot(old_stps[i], old_dirs[i])
        

      # iteration in L-BFGS loop collapsed to use just one buffer
      # need to be accessed element-by-element, so don't re-type tensor:
      al = [0]*nCorrection

      q = -g
      for i in range(k-1, -1, -1):
        al[i] = dot(old_dirs[i], q) * ro[i]
        q = q - al[i]*old_stps[i]

      # multiply by initial Hessian
      r = q*Hdiag
      for i in range(k):
        be_i = dot(old_stps[i], r) * ro[i]
        r += (al[i]-be_i)*old_dirs[i]
        
      d = r
      # final direction is in r/d (same object)

    g_old = g
    f_old = f
    
    ############################################################
    ## compute step length
    ############################################################
    # directional derivative
    gtd = dot(g, d)

    # check that progress can be made along that direction
    if gtd > -tolX:
      verbose("Can not make progress along direction.")
      break

    # reset initial guess for step size
    if state.nIter == 1:
      tmp1 = tf.abs(g)
      t = min(1, 1/tf.reduce_sum(tmp1))
    else:
      t = learningRate


    # optional line search: user function
    lsFuncEval = 0
    if lineSearch and isinstance(lineSearch) == types.FunctionType:
      # perform line search, using user function
      f,g,x,t,lsFuncEval = lineSearch(opfunc,x,t,d,f,g,gtd,lineSearchOpts)
      f_hist.append(f)
    else:
      # no line search, simply move with fixed-step
      x += t*d
      
      if nIter != maxIter:
        # re-evaluate function only if not in last iteration
        # the reason we do this: in a stochastic setting,
        # no use to re-evaluate that function here
        f, g = opfunc(x)
        
        lsFuncEval = 1
        f_hist.append(f)


    # update func eval
    currentFuncEval = currentFuncEval + lsFuncEval
    state.funcEval = state.funcEval + lsFuncEval

    ############################################################
    ## check conditions
    ############################################################
    if nIter == maxIter:
      break

    if currentFuncEval >= maxEval:
      # max nb of function evals
      verbose('max nb of function evals')
      break

    tmp1 = tf.abs(g)
    if tf.reduce_sum(tmp1) <=tolFun:
      # check optimality
      verbose('optimality condition below tolFun')
      break
    
    tmp1 = tf.abs(d*t)
    if tf.reduce_sum(tmp1) <= tolX:
      # step size below tolX
      verbose('step size below tolX')
      break

    if tf.abs(f-f_old) < tolX:
      # function value changing less than tolX
      verbose('function value changing less than tolX'+str(tf.abs(f-f_old)))
      break

    if do_verbose:
      print("Step %3d loss %6.5f msec %6.3f"%(nIter, f.numpy(), u.last_time()))
      u.record_time()
      times.append(u.last_time())

    if nIter == maxIter - 1:
      final_loss = f.numpy()


  # save state
  state.old_dirs = old_dirs
  state.old_stps = old_stps
  state.Hdiag = Hdiag
  state.g_old = g_old
  state.f_old = f_old
  state.t = t
  state.d = d

  return x, f_hist, currentFuncEval

# dummy/Struct gives Lua-like struct object with 0 defaults
class dummy(object):
  pass

class Struct(dummy):
  def __getattribute__(self, key):
    if key == '__dict__':
      return super(dummy, self).__getattribute__('__dict__')
    return self.__dict__.get(key, 0)

W_flat = None
def benchmark(batch_size, iters, seed=1, cuda=True, history=100, verbose=False):
  global final_loss, W_flat
  tf.set_random_seed(seed)
  np.random.seed(seed)
  
  images = tf.constant(u.get_mnist_images(batch_size).T)
  images = images[:batch_size]
  if cuda:
    images = images.gpu()
  data = images

  if cuda:
    device='/gpu:0'
  else:
    device=''

  device_ctx = tf.device(device)
  device_ctx.__enter__()

  visible_size = 28*28
  hidden_size = 196
  initial_val = tf.zeros([visible_size*hidden_size])
  if W_flat is None:
    W_flat = tfe.Variable(initial_val, name='W_flat')
  W_flat.assign(initial_val)
  
  
  def loss_fn(w_flat):
    w = tf.reshape(w_flat, [visible_size, hidden_size])
    x = tf.matmul(data, w)
    x = tf.sigmoid(x)
    x = tf.matmul(x, w, transpose_b=True)
    x = tf.sigmoid(x)
    return tf.reduce_mean(tf.square(x-data))

  value_and_gradients_fn = tfe.value_and_gradients_function(loss_fn)
  def opfunc(x):  # returns (value, gradient)
    value, grads = value_and_gradients_fn(x)
    return value, grads[0]
      
  # initialize weights
  W_flat.assign(u.ng_init(visible_size, hidden_size).flatten())

  state = Struct()
  config = Struct()
  config.maxIter = iters
  config.nCorrection = history
  config.verbose = True
  x, f_hist, currentFuncEval = lbfgs(opfunc, W_flat, config, state, verbose)

  if verbose:
    u.summarize_time()

  s = ','.join(["%f"%(n,) for n in times[2:]])
  print('{', s,'}')
  
  return final_loss


def main():
  import common_gd
  args = common_gd.args
  args.cuda = not args.no_cuda and (tfe.num_gpus() > 0)

  print(benchmark(batch_size=args.batch_size, iters=args.iters, seed=args.seed, cuda=args.cuda, history=args.history, verbose=True))

if __name__=='__main__':
  main()
  
