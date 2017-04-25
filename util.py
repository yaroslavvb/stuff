import tensorflow as tf
import networkx as nx
import numpy as np
import os
import sys
import tensorflow as tf
import time
import traceback
import inspect

default_dtype = tf.float32
USE_MKL_SVD=True                   # Tensorflow vs MKL SVD

if USE_MKL_SVD:
  assert np.__config__.get_info("lapack_mkl_info"), "No MKL detected :("


from scipy import linalg

# TODO: speed-up tests by reusing session

def concat_blocks(blocks, validate_dims=True):
  """Takes 2d grid of blocks representing matrices and concatenates to single
  matrix (aka ArrayFlatten)"""

  if validate_dims:
    col_dims = np.array([[int(b.shape[1]) for b in row] for row in blocks])
    col_sums = col_dims.sum(1)
    assert (col_sums[0] == col_sums).all()
    row_dims = np.array([[int(b.shape[0]) for b in row] for row in blocks])
    row_sums = row_dims.sum(0)
    assert (row_sums[0] == row_sums).all()
  
  block_rows = [tf.concat(row, axis=1) for row in blocks]
  return tf.concat(block_rows, axis=0)

def concat_blocks_test():
  blocks = [[tf.constant([[1]]), tf.constant([[1,2]])],
            [tf.transpose(tf.constant([[1,2]])), tf.constant([[1,2],[3,4]])]]
  result = concat_blocks(blocks)
  sess = tf.Session()
  result0 = sess.run(result)
  check_equal(result0, [[1, 1, 2], [1, 1, 2], [2, 3, 4]])


def partition_matrix_evenly(mat, splits):
  """Breaks matrix into 2d grid of equal size."""
  assert int(mat.shape[0])%splits==0
  assert int(mat.shape[1])%splits==0
  
  row_chunks = tf.split(mat, splits, axis=0)
  col_chunks = [tf.split(chunk, splits, axis=1) for chunk in row_chunks]
  return col_chunks

def partition_matrix_evenly_test():
  a = tf.reshape([1,2,3,4], (2,2))
  blocks = partition_matrix_evenly(a, 2)
  a2 = concat_blocks(blocks)
  sess = tf.Session()
  check_equal(sess.run(a2), sess.run(a))

# inverse of concat blocks
def partition_matrix(mat, sizes):
  pass

def partition_matrix_test():
  pass


  # TODO: add name property
def pseudo_inverse(mat, eps=1e-10):
  """Computes pseudo-inverse of mat, treating eigenvalues below eps as 0."""
  
  s, u, v = tf.svd(mat)
  eps = 1e-10   # zero threshold for eigenvalues
  si = tf.where(tf.less(s, eps), s, 1./s)
  return u @ tf.diag(si) @ tf.transpose(v)

def symsqrt(mat, eps=1e-7):
  """Symmetric square root."""
  s, u, v = tf.svd(mat)
  # sqrt is unstable around 0, just use 0 in such case
  print("Warning, cutting off at eps")
  si = tf.where(tf.less(s, eps), s, tf.sqrt(s))
  return u @ tf.diag(si) @ tf.transpose(v)

def pseudo_inverse_sqrt(mat, eps=1e-7):
  """half pseduo-inverse"""
  s, u, v = tf.svd(mat)
  # zero threshold for eigenvalues
  si = tf.where(tf.less(s, eps), s, 1./tf.sqrt(s))
  return u @ tf.diag(si) @ tf.transpose(v)

def pseudo_inverse_sqrt2(svd, eps=1e-7):
  """half pseduo-inverse, accepting existing values"""
  # zero threshold for eigenvalues
  if svd.__class__.__name__=='SvdTuple':
    (s, u, v) = (svd.s, svd.u, svd.v)
  elif svd.__class__.__name__=='SvdWrapper':
    (s, u, v) = (svd.s, svd.u, svd.v)
  else:
    assert False, "Unknown type"
  si = tf.where(tf.less(s, eps), s, 1./tf.sqrt(s))
  return u @ tf.diag(si) @ tf.transpose(v)

def pseudo_inverse2(svd, eps=1e-7):
  """pseudo-inverse, accepting existing values"""
  # use float32 machine precision as cut-off (works for MKL)
  # https://www.wolframcloud.com/objects/927b2aa5-de9c-46f5-89fe-c4a58aa4c04b
  if svd.__class__.__name__=='SvdTuple':
    (s, u, v) = (svd.s, svd.u, svd.v)
  elif svd.__class__.__name__=='SvdWrapper':
    (s, u, v) = (svd.s, svd.u, svd.v)
  else:
    assert False, "Unknown type"
  max_eigen = tf.reduce_max(s)
  si = tf.where(s/max_eigen<eps, 0.*s, 1./s)
  return u @ tf.diag(si) @ tf.transpose(v)

def pseudo_inverse_stable(svd, eps=1e-7):
  """pseudo-inverse, accepting existing values"""
  # use float32 machine precision as cut-off (works for MKL)
  # https://www.wolframcloud.com/objects/927b2aa5-de9c-46f5-89fe-c4a58aa4c04b
  if svd.__class__.__name__=='SvdTuple':
    (s, u, v) = (svd.s, svd.u, svd.v)
  elif svd.__class__.__name__=='SvdWrapper':
    (s, u, v) = (svd.s, svd.u, svd.v)
  else:
    assert False, "Unknown type"
  max_eigen = tf.reduce_max(s)
  si = tf.where(s/max_eigen<eps, 0.*s, tf.pow(s, -0.9))
  return u @ tf.diag(si) @ tf.transpose(v)

# todo: rename l to L
def regularized_inverse(mat, l=0.1):
  return tf.matrix_inverse(mat + l*Identity(int(mat.shape[0])))

def regularized_inverse2(svd, L=1e-3):
  """Regularized inverse, working from SVD"""
  if svd.__class__.__name__=='SvdTuple' or svd.__class__.__name__=='SvdWrapper':
    (s, u, v) = (svd.s, svd.u, svd.v)
  else:
    assert False, "Unknown type"
  max_eigen = tf.reduce_max(s)
  #  max_eigen = tf.Print(max_eigen, [max_eigen], "max_eigen")
  #si = 1/(s + L*tf.ones_like(s)/max_eigen)
  si = 1/(s+L*tf.ones_like(s))
  return u @ tf.diag(si) @ tf.transpose(v)

def pseudo_inverse_scipy(tensor):
    dtype = tensor.dtype
    print(linalg.pinv, tensor, dtype)
    result = tf.py_func(linalg.pinv, [tensor],
                        [dtype])[0]
    result.set_shape(tensor.shape)
    return result
  
  

def Identity(n, dtype=None, name=None):
  """Identity matrix of size n."""
  if hasattr(n, "shape"):  # got a Tensor
    nn = fix_shape(n.shape)
    assert nn[0] == nn[1]
    n = nn[0]
  if not dtype:
    dtype = default_dtype
  return tf.diag(tf.ones((n,), dtype=dtype), name=name)

def ones(n, dtype=None, name=None):
  if not dtype:
    dtype = default_dtype
  return tf.ones((n,), dtype=dtype, name=name)

# partitions numpy array into sublists of given sizes
def partition_list_np(vec, sizes):
  assert np.sum(sizes) == len(vec)
  splits = []
  current_idx = 0
  for i in range(len(sizes)):
    splits.append(vec[current_idx: current_idx+sizes[i]])
    current_idx += sizes[i]
  assert current_idx == len(vec)
  return splits

def chunks(l, n):
  """Yield successive n-sized chunks from l."""
  for i in range(0, len(l), n):
    yield l[i:i + n]

def partition_list(l, sizes):
  """Partition l into sublists of given sizes."""
  assert len(l.shape) == 1
  assert np.sum(sizes) == l.shape[0]
  splits = []
  current_idx = 0
  for i in range(len(sizes)):
    splits.append(l[current_idx: current_idx+sizes[i]])
    current_idx += sizes[i]
  return splits

def partition_list_test():
  vec = tf.constant([1,2,3,4,5])
  sess = tf.Session()
  result = sess.run(partition_list(vec, [3, 2]))
  check_equal(result[0], [1,2,3])
  assert (result[1] == [4,5]).all()


def v2c(vec):
  """Convert vector to column matrix."""
  assert len(vec.shape) == 1
  return tf.expand_dims(vec, 1)

def v2c_np(vec):
  """Convert vector to column matrix."""
  assert len(vec.shape) == 1
  return np.expand_dims(vec, 1)

def v2r(vec):
  """Convert vector into row matrix."""
  assert len(vec.shape) == 1
  return tf.expand_dims(vec, 0)
  
def c2v(col):
  """Convert vector into row matrix."""
  assert len(col.shape) == 2
  assert col.shape[1] == 1
  return tf.reshape(col, [-1])


def unvectorize_np(vec, rows):
  """Turn vectorized version of tensor into original matrix with given
  number of rows."""
  assert len(vec)%rows==0
  cols = len(vec)//rows;
  return np.array(np.split(vec, cols)).T

def unvec(vec, rows):
  """Turn vectorized version of tensor into original matrix with given
  number of rows."""
  assert len(vec.shape) == 1
  assert vec.shape[0]%rows == 0
  cols = int(vec.shape[0]//rows)
  return tf.transpose(tf.reshape(vec, (cols, -1)))
#  cols = [v2r(v) for v in tf.split(vec, cols)]
#  return tf.transpose(tf.concat(cols, 0))

def unvec_test():
  vec = tf.constant([1,2,3,4,5,6])
  sess = tf.Session()
  result = sess.run(unvec(vec, 2))
  assert (result==[[1,3,5],[2,4,6]]).all()

def vectorize_np(mat):
  return mat.reshape((-1, 1), order="F")

def vec(mat):
  """Vectorize matrix."""
  return tf.reshape(tf.transpose(mat), [-1,1])

def vec_test():
  mat = tf.constant([[1, 3, 5], [2, 4, 6]])
  sess = tf.Session()
  check_equal(sess.run(c2v(vec(mat))), [1,2,3,4,5,6])


def Kmat(rows, cols):
  """Commutation matrix. Kmat(a,b).vec(M) takes vec of a,b matrix M to vec of
  its transpose."""
  input_mat = np.reshape(np.arange(rows*cols),[rows,-1]).astype(np.int32)
  output_mat = input_mat.T
    
  input_vec = vectorize_np(input_mat)
  output_vec = vectorize_np(output_mat)
    
  K = np.zeros((rows*cols, rows*cols), dtype=np.int32)
  for output_idx in range(rows*cols):
    for input_idx in range(rows*cols):
      K[output_idx, input_idx] = (output_vec[output_idx] == input_vec[input_idx])
  return K

def Kmat_test():
  check_equal(Kmat(3,2),
              [[1, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0],
               [0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 1]])

  check_equal(Kmat(2,3),
              [[1, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 1, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 1]])

# turns flattened representation into list of matrices with given matrix
# sizes
def unflatten_np(Wf, fs):
  if len(Wf.shape)==2 and Wf.shape[1] == 1:  # treat col mats as vectors
    Wf = Wf.reshape(-1)
    
  dims = [(fs[i+1],fs[i]) for i in range(len(fs)-1)]
  sizes = [s[0]*s[1] for s in dims]
  assert np.sum(sizes)==len(Wf)
  Wsf = partition_list_np(Wf, sizes)
  Ws = [unvectorize_np(Wsf[i], dims[i][0]) for i in range(len(sizes))]
  return Ws

def flatten_np(Ws):
  return np.concatenate([np.reshape(vectorize_np(W),(-1,)) for W in Ws],
                          axis=0)
def flatten_np_test():
  vec = np.asarray(range(1, 11))
  fs = [2,2,2,1]
  result = unflatten_np(vec, fs)
  result2 = flatten_np(result)
  check_equal(vec, result2)

def unflatten(Wf, fs):
  """Turn flattened Tensor into list of rank-2 tensors with given sizes."""
  
  Wf_shape = fix_shape(Wf.shape)
  if len(Wf_shape)==2 and Wf_shape[1] == 1:  # treat col mats as vectors
    Wf = tf.reshape(Wf, [-1])
  dims = [(fs[i+1],fs[i]) for i in range(len(fs)-1)]
  sizes = [s[0]*s[1] for s in dims]
  assert len(Wf.shape) == 1
  assert np.sum(sizes)==Wf.shape[0]
  Wsf = partition_list(Wf, sizes)
  Ws = [unvec(Wsf[i], dims[i][0]) for i in range(len(sizes))]
  return Ws

def unflatten_test():
  vec = tf.constant(list(range(1, 11)))
  sess = tf.Session()
  fs = [2,2,2,1]
  result = sess.run(unflatten(vec, fs))
  check_equal(result[0], [[1,3],[2,4]])
  check_equal(result[1], [[5,7],[6,8]])
  check_equal(result[2], [[9, 10]])

def flatten(Ws):
  """Inverse of unflatten."""
  return tf.concat([tf.reshape(vec(W),(-1,)) for W in Ws], axis=0)

def flatten_test():
  vec = tf.constant(list(range(1, 11)))
  sess = tf.Session()
  fs = [2,2,2,1]
  result = unflatten(vec, fs)
  result2 = flatten(result)
  check_equal(sess.run(vec), sess.run(result2))

def check_close(a0, b0):
  return check_equal(a0, b0, rtol=1e-5, atol=1e-9)
  
def check_equal(a0, b0, rtol=1e-9, atol=1e-12):
  """Helper function to check that two vectors are equal."""

  a = a0.eval() if hasattr(a0, "eval") else a0
  b = b0.eval() if hasattr(b0, "eval") else b0
  
  try:
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
  except Exception as e:
    print("Error" + "-"*60)
    for line in traceback.format_stack():
      print(line.strip())
        
    exc_type, exc_value, exc_traceback = sys.exc_info()
    print("*** print_tb:")
    traceback.print_tb(exc_traceback, limit=10, file=sys.stdout)
    efmt = traceback.format_exc()
    print(efmt)
    #    import pdb; pdb.set_trace()

# TensorShape([Dimension(2), Dimension(10)]) => (2, 10)
def fix_shape(tf_shape):
  return tuple(int(dim) for dim in tf_shape)

def kronecker_cols(a, b):
  """Treats rank-1 vectors a, b as columns, returns Kronecker product a x b."""
  
  assert len(a.get_shape())==1, "Input a must be rank-1, got shape %s" %(a.get_shape(),)
  assert len(b.get_shape())==1, "Input b must be rank-1, got shape %s"%(a.get_shape(),)
  segments = []
  for i in range(a.get_shape()[0]):
    segments.append(a[i]*b)
  result_vec = tf.concat(segments, axis=0)
  result_col = tf.expand_dims(result_vec, 1)
  return result_col

def kronecker_cols_test():
  a = tf.constant([1,2])
  b = tf.constant([3,4])
  c = tf.transpose(tf.constant([[3,4,6,8]]))
  sess = tf.Session()
  assert sess.run(tf.equal(kronecker_cols(a, b), c)).all()


def kronecker(A, B, do_shape_inference=True):
  """Kronecker product of A,B.
  turn_off_shape_inference: if True, makes 10x10 kron go 2.4 sec -> 0.9 sec
  """

  Arows, Acols = fix_shape(A.shape)
  Brows, Bcols = fix_shape(B.shape)
  Crows, Ccols = Arows*Brows, Acols*Bcols
  
  temp = tf.reshape(A, [-1, 1, 1])*tf.expand_dims(B, 0)
  Bshape = tf.constant((Brows, Bcols))

  # turn off shape inference
  if not do_shape_inference:
    disable_shape_inference()

  # [1, n, m] => [n, m]
  slices = [tf.reshape(s, Bshape) for s in tf.split(temp, Crows)]
  
  #  import pdb; pdb.set_trace()
  grid = list(chunks(slices, Acols))
  assert len(grid) == Arows
  result = concat_blocks(grid, validate_dims=do_shape_inference)

  if not do_shape_inference:
    enable_shape_inference()
    result.set_shape((Arows*Brows, Acols*Bcols))
    
  return result

kr = kronecker

def kronecker_test():
  A0 = [[1,2],[3,4]]
  B0 = [[6,7],[8,9]]
  A = tf.constant(A0)
  B = tf.constant(B0)
  C = kronecker(A, B)
  sess = tf.Session()
  C0 = sess.run(C)
  Ct = [[6, 7, 12, 14], [8, 9, 16, 18], [18, 21, 24, 28], [24, 27, 32, 36]]
  Cnp = np.kron(A0, B0)
  check_equal(C0, Ct)
  check_equal(C0, Cnp)


def col(A,i):
  """Extracts i'th column of matrix A"""
  assert len(A.get_shape())==2
  assert i>=0 and i < A.get_shape()[1]
  return tf.expand_dims(A[:,i], 1)


def khatri_rao(A, B):
  Arows, Acols = fix_shape(A.shape)
  Brows, Bcols = fix_shape(B.shape)
  assert Acols==Bcols
  return tf.reshape(tf.einsum("ik,jk->ijk", A, B), (Arows*Brows, Acols))


def khatri_rao_test():
  A = tf.constant([[1, 2], [3, 4]])
  B = tf.constant([[5, 6], [7, 8]])
  C = tf.constant([[5,12], [7,16], [15,24], [21,32]])
  sess = tf.Session()
  assert sess.run(tf.equal(khatri_rao(A, B), C)).all()

  
def relu_mask(a, dtype=default_dtype):
  """Produces mask of 1s for positive values and 0s for negative values."""
  from tensorflow.python.ops import gen_nn_ops
  ones = tf.ones(a.get_shape(), dtype=dtype)
  return gen_nn_ops._relu_grad(ones, a)

def relu_mask_test():
  a = tf.constant([-1,0,1,2], dtype=default_dtype)
  sess = tf.Session()
  check_equal(sess.run(relu_mask(a)), [0,0,1,1])

def assert_rectangular(blocks):
  lengths = np.array([len(row) for row in blocks])
  assert (lengths==lengths[0]).all()
  
def empty_grid(rows, cols):
  """Create empty list of lists of rows-by-cols shape."""
  result = []
  for i in range(rows):
    result.append([None]*cols)
  return result

def block_diagonal_inverse(blocks):
  """Invert diagonal blocks, leave remaining unchanged."""
  
  assert_rectangular(blocks)
  num_rows = len(blocks)
  num_cols = len(blocks[0])

  result = empty_grid(num_rows, num_cols)
  dtype = blocks[0][0].dtype   # TODO: assert same dtype
  
  for i in range(len(blocks)):
    for j in range(len(blocks[0])):
      block = blocks[i][j]
      if i == j:
        result[i][j] = pseudo_inverse(block)
      else:
        result[i][j] = tf.zeros(shape=block.get_shape(),
                                dtype=dtype)
  return result
        
def block_diagonal_inverse_sqrt(blocks):
  assert_rectangular(blocks)
  num_rows = len(blocks)
  num_cols = len(blocks[0])

  result = empty_grid(num_rows, num_cols)
  dtype = blocks[0][0].dtype   # TODO: assert same dtype
  
  for i in range(len(blocks)):
    for j in range(len(blocks[0])):
      block = blocks[i][j]
      if i == j:
        result[i][j] = pseudo_inverse_sqrt(block)
      else:
        result[i][j] = tf.zeros(shape=block.get_shape(),
                                dtype=dtype)
  return result


def block_diagonal_inverse_test():
  sess = tf.Session()
  blocks = [[2*Identity(3), tf.ones((3, 1))],
              [tf.ones((1,3)), 2*Identity(1)]]
  new_blocks = block_diagonal_inverse(blocks)
  actual = concat_blocks(new_blocks)
  expected = 0.5*Identity(4)
  check_equal(sess.run(actual), sess.run(expected))

  
def t(x):
  return tf.transpose(x)

  
# Time tracking functions
global_time_list = []
global_last_time = 0
def reset_time():
  global global_time_list, global_last_time
  global_time_list = []
  global_last_time = time.time()
  
def record_time():
  global global_last_time, global_time_list
  new_time = time.time()
  global_time_list.append(new_time - global_last_time)
  global_last_time = time.time()

def last_time():
  global global_last_time, global_time_list
  if global_time_list:
    return 1000*global_time_list[-1]
  else:
    return 0

def summarize_time(time_list=None):
  if time_list is None:
    time_list = global_time_list
    
  time_list = 1000*np.array(time_list)  # get seconds, convert to ms
  if len(time_list)>0:
    min = np.min(time_list)
    median = np.median(time_list)
    formatted = ["%.2f"%(d,) for d in time_list[:10]]
    print("Times: min: %.2f, median: %.2f, times: %s"%(min, median,",".join(formatted)))
  else:
    print("Times: <empty>")
    
def summarize_graph(g=None):
  if not g:
    g = tf.get_default_graph()
  print("Graph: %d ops, %d MBs"%(len(g.get_operations()),
                                 len(str(g.as_graph_def()))/10**6))

from tensorflow.python.framework import ops
original_shape_func = ops.set_shapes_for_outputs
def disable_shape_inference():
  ops.set_shapes_for_outputs = lambda _: _
  
def enable_shape_inference():
  ops.set_shapes_for_outputs = original_shape_func

def run_all_tests(module):
  all_functions = inspect.getmembers(module, inspect.isfunction)
  for name,func in all_functions:
    if name.endswith("_test"):
      print("Testing "+name)
      with timeit():
        func()
  print(module.__name__+" tests passed.")

def dump(result, fname):
  """Save result to file."""
  result = result.eval() if hasattr(result, "eval") else result
  result = np.asarray(result)
  if result.shape == ():   # savetxt has problems with scalars
    result = np.expand_dims(result, 0)
  location = os.getcwd()+"/data/"+fname
  # special handling for integer datatypes
  if (
      result.dtype == np.uint8 or result.dtype == np.int8 or
      result.dtype == np.uint16 or result.dtype == np.int16 or
      result.dtype == np.uint32 or result.dtype == np.int32 or
      result.dtype == np.uint64 or result.dtype == np.int64
  ):
    np.savetxt(location, result, fmt="%d", delimiter=',')
  else:
    np.savetxt(location, result, delimiter=',')
  print(location)


def frobenius_np(a):
  return np.sqrt(np.sum(np.square(a)))

def nan_check(result):
  result = result.eval() if hasattr(result, "eval") else result
  result = np.asarray(result)
  print("result any NaNs: %s"% (np.isnan(result).any(),))


def L2(t):
  """Squared L2 norm of t."""
  return tf.reduce_sum(tf.square(t))

class timeit:    
  def __enter__(self):
    self.start = time.clock()
    return self

  def __exit__(self, *args):
    self.end = time.clock()
    self.interval = self.end - self.start
    print("Elapsed: %.2f ms"%(self.interval*1000))


# graph traversal
# computation flows from parents to children
# to find path from target to dependency, do
# nx.shortest_path(gg, dependency, target)
def parents(op): return set(input.op for input in op.inputs)
def children(op): return set(op for out in op.outputs for op in out.consumers())
def dict_graph():
  """Creates dictionary {node: {child1, child2, ..},..} for current
  TensorFlow graph. Result is compatible with networkx/toposort"""

  ops = tf.get_default_graph().get_operations()
  return {op: children(op) for op in ops}
def nx_graph():
  return nx.DiGraph(dict_graph())

def shortest_path(dep, target):
  if hasattr(dep, "op"):
    dep = dep.op
  if hasattr(target, "op"):
    target = target.op
  return nx.shortest_path(nx_graph(), dep, target)

def list_or_tuple(k):
  return isinstance(k, list) or isinstance(k, tuple)

def is_numeric(ndarray):
  ndarray = np.asarray(ndarray)
  return np.issubdtype(ndarray.dtype, np.number)

class VarInfo:
  """Encapsulate variable info."""
  def __init__(self, setter, p):
    self.setter = setter
    self.p = p

class SvdTuple:
  """Object to store svd tuple.
  Create as SvdTuple((s,u,v)) or SvdTuple(s, u, v).
  """
  def __init__(self, suv, *args):
    if list_or_tuple(suv):
      s, u, v = suv
    else:
      s = suv
      u = args[0]
      v = args[1]
      assert len(args) == 2
    self.s = s
    self.u = u
    self.v = v


class SvdWrapper:
  """Encapsulates variables needed to perform SVD of a TensorFlow target.
  Initialize: wrapper = SvdWrapper(tensorflow_var)
  Trigger SVD: wrapper.update_tf() or wrapper.update_scipy()
  Access result as TF vars: wrapper.s, wrapper.u, wrapper.v
  """
  
  def __init__(self, target, name):
    self.name = name
    self.target = target
    self.tf_svd = SvdTuple(tf.svd(target))

    self.init = SvdTuple(
      ones(target.shape[0], name=name+"_s_init"),
      Identity(target.shape[0], name=name+"_u_init"),
      Identity(target.shape[0], name=name+"_v_init")
    )

    assert self.tf_svd.s.shape == self.init.s.shape
    assert self.tf_svd.u.shape == self.init.u.shape
    assert self.tf_svd.v.shape == self.init.v.shape

    self.cached = SvdTuple(
      tf.Variable(self.init.s, name=name+"_s"),
      tf.Variable(self.init.u, name=name+"_u"),
      tf.Variable(self.init.v, name=name+"_v")
    )

    self.s = self.cached.s
    self.u = self.cached.u
    self.v = self.cached.v
    
    self.holder = SvdTuple(
      tf.placeholder(default_dtype, shape=self.cached.s.shape, name=name+"_s_holder"),
      tf.placeholder(default_dtype, shape=self.cached.u.shape, name=name+"_u_holder"),
      tf.placeholder(default_dtype, shape=self.cached.v.shape, name=name+"_v_holder")
    )

    self.update_tf_op = tf.group(
      self.cached.s.assign(self.tf_svd.s),
      self.cached.u.assign(self.tf_svd.u),
      self.cached.v.assign(self.tf_svd.v)
    )

    self.update_external_op = tf.group(
      self.cached.s.assign(self.holder.s),
      self.cached.u.assign(self.holder.u),
      self.cached.v.assign(self.holder.v)
    )

    self.init_ops = (self.s.initializer, self.u.initializer, self.v.initializer)
  

  def update(self):
    if USE_MKL_SVD:
      self.update_scipy()
    else:
      self.update_tf()
      
  def update_tf(self):
    sess = tf.get_default_session()
    sess.run(self.update_tf_op)
    
  def update_scipy(self):
    sess = tf.get_default_session()
    target0 = self.target.eval()
    # A=u.diag(s).v', singular vectors are columns
    # TODO: catch "ValueError: array must not contain infs or NaNs"
    u0, s0, vt0 = linalg.svd(target0)
    v0 = vt0.T
    #    v0 = vt0 # bug, makes loss increase, use for sanity checks
    feed_dict = {self.holder.u: u0,
                 self.holder.v: v0,
                 self.holder.s: s0}
    sess.run(self.update_external_op, feed_dict=feed_dict)

def extract_grad(grads_and_vars, var):
  if isinstance(var, str):
    varname = var
  else:
    varname = var.name
  vals = []
  for (grad, var) in grads_and_vars:
    if var.name == varname:
      vals.append(var)
  assert length(vals)==1
  return vals[0]

def intersept_op_creation(op_type_name_to_intercept):
  """Drops into PDB when particular op type is added to graph."""
  from tensorflow.python.framework import op_def_library
  old_apply_op = op_def_library.OpDefLibrary.apply_op
  def my_apply_op(obj, op_type_name, name=None, **keywords):
    print(op_type_name+"-"+str(name))
    if op_type_name == op_type_name_to_intercept:
      import pdb; pdb.set_trace()
    return(old_apply_op(obj, op_type_name, name=name, **keywords))
  op_def_library.OpDefLibrary.apply_op=my_apply_op

if __name__=='__main__':
  run_all_tests(sys.modules[__name__])
