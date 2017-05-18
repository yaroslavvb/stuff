"""Helpers to replicate computation specified as part of existing graph."""


# helper to allow @profile decorators even when no line profiler
import builtins
try:
    builtins.profile
except AttributeError:
    def profile(func): return func
    builtins.profile = profile

from collections import OrderedDict
from pprint import pprint
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import function
from tensorflow.python.ops import gen_math_ops
import inspect
import numpy as np
import contextlib
import os, sys
import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
import toposort
import whitening_util as u

class GraphTemplate:
  """Represents TensorFlow Graph that can be applied to new outputs.
  
  Example:

  Define some computation with a input, b output, apply it to new input (c)
  a = tf.ones((2,))
  b = tf.square(a)
  t = GraphTemplate([a], [b])
  c = tf.zeros((2,))
  
  [d] = t.apply([c])

  Restrictions on computation used to create template:
  - no control flow
  - no variables
  - no colocation constraints to nodes outside of template
  - do not span device boundaries (not enforced)

  Also note that:
  surrounding name_scope is ignored when instantiating template
  """
  
  def __init__(self, inputs, outputs, within_ops=None):
    assert isinstance(inputs, (list, tuple))
    assert isinstance(outputs, (list, tuple))

    g = tf.get_default_graph()
    self.g = g

    # special handling for Variable objects because graph_editor only
    # supports Tensor objects. Instead of replacing Variable with new inputs
    # replace all of its read endpoints.
    # 
    # Find all Identity ops connected to variable op, replace them all in
    # tandem on apply.
    # [tensor, tensor, variable] becomes
    # [tensor, tensor, (var_read1, var_read2)]

    new_inputs = []
    for input in inputs:
      if isinstance(input, tf.Variable):
        # find all the read endpoints of the variable
        read_tensors = []
        for consumer in input.op.outputs[0].consumers():
          if consumer.type == 'Identity':
            read_tensors.append(consumer.outputs[0])
        new_inputs.append(read_tensors)
      else:
        new_inputs.append(input)
    inputs = new_inputs
    
    self.inputs = inputs
    self.input_ops = [tensor.op for tensor in flatten1(inputs)]
    self.outputs = outputs

    # only support 1 output for now, need extra logic in apply
    for output_ts in outputs:
      num_siblings = len(output_ts.op.outputs)-1
      assert num_siblings == 0

    # obtain part of graph that's needed
    output_ops = [ts.op for ts in self.outputs]

    self.ops = ge.get_backward_walk_ops(output_ops,
                                        inclusive=True,
                                        stop_at_ts=flatten1(self.inputs),
                                        within_ops=within_ops)

    # workaround for https://github.com/tensorflow/tensorflow/issues/9978
    clear_original_ops(self.ops)

  @profile
  def apply(self, new_inputs, update_colocation_groups=True):
    assert len(new_inputs) == len(self.inputs)
    g = tf.get_default_graph()  # todo: make that member variable

    new_inputs2 = []
    # replace variable inputs with their read endpoint
    for input in new_inputs:
      if isinstance(input, tf.Variable):
        new_inputs2.append(input.read_value())
      else:
        new_inputs2.append(input)
    new_inputs = new_inputs2
    
    replacements = OrderedDict()
    for old_input_ts, new_input_ts in zip(self.inputs, new_inputs):
      # shape/dtype checks
      if isinstance(old_input_ts, (list, tuple)):
        reference_ts = old_input_ts[0]
      else:
        reference_ts = old_input_ts
      assert reference_ts.get_shape() == new_input_ts.get_shape()
      assert reference_ts.dtype == new_input_ts.dtype

      # Variable with multiple read endpoints, replace all of them with
      # new input tensor
      if isinstance(old_input_ts, (list, tuple)):
        for sub_input in old_input_ts:
          replacements[sub_input] = new_input_ts
      # regular Tensor
      else:
        replacements[old_input_ts] = new_input_ts


    # sanity checks
    # copying Variables is confusing because they don't get added
    # to GLOBAL_VARIABLES collection hence escape global initialization
    # therefore forbit it
    for op in self.ops:
      if op.type.startswith('Variable'): # 'VariableV2' or 'Variable'
        assert False, "Can't copy variables"


    # TODO: remove this
    def summarize_ts(ts):
      from collections import Counter
      type_counter = Counter()
      ops = set([tensor.op for tensor in ts])
      print Counter([op.type for op in ops]).most_common(10)



        
    sgv = ge.sgv(self.ops)
    #    import pdb; pdb.set_trace()
    copied_sgv, info = ge.copy_with_input_replacements(sgv,
                                                       replacements)


    # converting between Python bytes and unicode
    def to_bytes(s): return s.encode('ascii')
    def from_bytes(s): return s.decode('ascii')

    # fix colocation constraints to point to copied ops
    # see https://github.com/tensorflow/tensorflow/issues/9925
    if update_colocation_groups:
      new_ops = [info._transformed_ops[op] for op in self.ops]
      for new_op in new_ops:
        assert len(new_op.colocation_groups()) == 1
        colocation_group = new_op.colocation_groups()[0]
        assert colocation_group.startswith(b'loc:@')
        colocated_with_name = from_bytes(colocation_group[len(b'loc:@'):])

        # if there were no colocation constraints, the op gets colocated with
        # itself (default colocation group), ignore that constraint
        if colocated_with_name == new_op.name:
          continue

        colocation_op = g.get_operation_by_name(colocated_with_name)
        if colocation_op in info._transformed_ops:
          new_colocation_op = info._transformed_ops[colocation_op]
        else:
          assert colocation_op in self.input_ops
          colocation_op_idx = self.input_ops.index(colocation_op)
          new_colocation_op = new_inputs[colocation_op_idx].op

        # overwrite existing _class attribute with new colocation constraints
        new_colocation_groups = [b'loc:@'+to_bytes(new_colocation_op.name)]
        new_op.node_def.attr["_class"].CopyFrom(attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(s=new_colocation_groups)))
    
    # place new ops on device from current device context
    device = get_current_device()
    if device:
      for op in info._transformed_ops.values():
        op._set_device(device)
      
    new_outputs = []
    for old_output_ts in self.outputs:
      new_output_op = info._transformed_ops[old_output_ts.op]
      new_output_ts = new_output_op.outputs[0]
      new_outputs.append(new_output_ts)
      
    return new_outputs


def clear_original_ops(ops):
  for op in ops:
    op._original_op = None

    
def tf_ops_to_graph(ops):
  """Creates op->children dictionary from list of TensorFlow ops."""

  def flatten(l): return [item for sublist in l for item in sublist]
  def children(op): return flatten(tensor.consumers() for tensor in op.outputs)
  return {op: set(children(op)) for op in ops}


def ops_in_toposorted_order(ops):
  """Produces ops in deterministic order such that children are executed
  after parents"""

  graph_dict = tf_ops_to_graph(ops)
  toposorted = toposort.toposort(graph_dict)
  ops = []
  # toposort assumes children are dependencies, reverse order
  for op_set in reversed(list(toposorted)):
    ops.extend(sorted(op_set, key=lambda op: op.name))
  return ops

class _DeviceCaptureOp(object):
  def __init__(self):
    self.device = None
  def _set_device(self, device):
    self.device = device

def get_current_device():
  """Returns device string of current graph context."""
  
  g = tf.get_default_graph()
  op = _DeviceCaptureOp()
  g._apply_device_functions(op)
  return op.device

def flatten1(list_of_lists):
  """Flattens list going down at most 1 level."""
  
  new_list = []
  for l in list_of_lists:
    if isinstance(l, (list, tuple)):
      new_list.extend(l)
    else:
      new_list.append(l)

  return new_list


def count_gpus():
  from tensorflow.python.client import device_lib
  count = 0
  for device in device_lib.list_local_devices():
    if device.device_type == "GPU":
      count+=1
  return count

def current_function_name():
  import inspect
  return inspect.stack()[1][0].f_code.co_name



def check_equal(a, b):
  a = np.asarray(a)
  b = np.asarray(b)
  np.testing.assert_allclose(a, b)


@contextlib.contextmanager
def capture_ops():
  """Captures any ops added to the default graph within this block."""
  #outer_graph = tf.get_default_graph()
  from tensorflow.python.framework import ops
  old_create_op =  ops.Graph.create_op
  op_list = []
  def new_create_op(graph_object, op_type, inputs, dtypes, input_types=None, name=None, attrs=None, op_def=None, compute_shapes=True, compute_device=True):
    # todo: remove keyword args
    op = old_create_op(graph_object, op_type=op_type, inputs=inputs, dtypes=dtypes, input_types=input_types, name=name, attrs=attrs, op_def=op_def, compute_shapes=compute_shapes, compute_device=compute_device)
    op_list.append(op)
    return op
  ops.Graph.create_op = new_create_op
  yield op_list

def capture_ops_test():
  a = tf.ones((), name="a")
  with capture_ops() as captured:
    b = tf.ones((), name="b")
  assert [op.name for op in captured] == ["b"]

def graph_template_test():
  tf.reset_default_graph()
  a = tf.ones((2,))
  b = tf.square(a)

  t = GraphTemplate([a], [b])
  c = tf.zeros((2,))
  
  [d] = t.apply([c])
  sess = tf.InteractiveSession()
  np.testing.assert_equal(sess.run(d), [0, 0])


def graph_devices_test():
  tf.reset_default_graph()
  with tf.device("/cpu:0"):
    a = tf.ones((2,))
    b = tf.square(a)

  t = GraphTemplate([a], [b])
  
  with tf.device("/cpu:1"):
    c = tf.zeros((2,))
    [d] = t.apply([c])
  graph_def_str = str(tf.get_default_graph().as_graph_def())
  assert 'cpu:1' in graph_def_str.lower()


def variables_test():
  a = tf.Variable(1)
  b = tf.square(a)
  c = tf.Variable(2)
  t = GraphTemplate([a], [b])
  [d] = t.apply([c])
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  assert d.eval() == 4

def multi_variables_test():
  tf.reset_default_graph()
  a = tf.Variable(1)
  b = a.read_value()
  c = a.read_value()
  d = b+c
  t = GraphTemplate([a], [d])
  e = tf.Variable(2)
  [f] = t.apply([e])
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  assert d.eval() == 2
  assert f.eval() == 4


def colocate_test():
  g = tf.get_default_graph()
  with tf.device('/cpu:0'):
    a = tf.ones((), name='a')
    with tf.get_default_graph().colocate_with(a):
      b = tf.add(a, 1, name='b')
  t = GraphTemplate([a], [b])
  [b2] = t.apply([tf.zeros(())])
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  assert b2.eval() == 1

  # make sure colocation is with new op (named zeros)
  assert b2.op.colocation_groups()[0] == b'loc:@zeros'


def between_devices_copy_test():
  g = tf.get_default_graph()
  config = tf.ConfigProto(device_count={"CPU": 2},
                          inter_op_parallelism_threads=2,
                          intra_op_parallelism_threads=1)
  sess = tf.InteractiveSession(config=config)

  with tf.device('/cpu:0'):
    a1 = tf.ones(())
    with g.colocate_with(a1):
      b1 = tf.square(a1)

  t = GraphTemplate([a1], [b1])

  with tf.device('/cpu:1'):
    a2 = tf.zeros(())
    [b2] = t.apply([a2])
  
  assert b2.device.lower().endswith('cpu:1')


def optimization_test():
  from tensorflow.python.ops import gen_math_ops
  def fast_sum(tensor, name=None):
    return gen_math_ops._sum(input=tensor,
                             reduction_indices=[],
                             keep_dims=False,
                             name=name)
  tf.reset_default_graph()
  g = tf.get_default_graph()
  config = tf.ConfigProto(device_count={"CPU": 2},
                          inter_op_parallelism_threads=2,
                          intra_op_parallelism_threads=1)
  sess = tf.InteractiveSession(config=config)
  params1 = tf.Variable(1, dtype=np.float32, name="params")

  temp = fast_sum(params1, name="sum_temp")
  cost1 = tf.square(temp, name="cost1")
  gradients1 = tf.gradients([cost1], [params1])
    
  templ = GraphTemplate([params1, cost1], gradients1)
  

  params2 = tf.Variable(1, dtype=np.float32, name="params2")
  cost2 = tf.square(fast_sum(params2))
  gradients2 = templ.apply([params2, cost2])
  train_op2 = params2.assign_sub(0.5*gradients2[0])

  sess.run(tf.global_variables_initializer())

  sess.run(train_op2)
  (cost1_, cost2_, params1_, params2_) = sess.run([cost1, cost2, params1,
                                                   params2])
  assert cost1_ == 1.0
  assert cost2_ == 0.0
  assert params1_ == 1.0
  assert params2_ == 0.0


def multidevice_shared_params_test():
  if count_gpus() < 2:
    print("Not enough GPUs, skipping %s"%(current_function_name()))
    return
    
  inputs = []
  params = tf.Variable(1, dtype=np.float32, name="params")
  costs = []
  grads = []
  for i in range(2):
    with tf.device("/gpu:%d"%(i,)):
      if i == 0:
        x = tf.zeros((), dtype=np.float32)
        cost = tf.square(tf.reduce_sum(params-x), name="cost")
        [gradient] = tf.gradients([cost], [params])
        templ = GraphTemplate([params, x, cost], [gradient])
        inputs.append(x)
        costs.append(cost)
        grads.append(gradient)
      else:
        x = tf.zeros((), dtype=np.float32)
        cost = tf.square(tf.reduce_sum(params-x), name="cost")
        [grad] = templ.apply([params, x, cost])
        inputs.append(x)
        costs.append(cost)
        grads.append(grad)
        
  # make train op
  total_grad = tf.add_n(grads)
  train_op = params.assign_sub(0.25*total_grad)
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())
  sess.run(train_op)
  costs0 = np.asarray(sess.run(costs))
  assert np.linalg.norm(costs0-[0, 0])<1e-7

def multidevice_separate_params_test():
  if count_gpus() < 2:
    print("Not enough GPUs, skipping %s"%(current_function_name()))
    return
  
  inputs = []
  params_list = []
  costs = []
  grads = []
  train_ops = []
  for i in range(2):
    with tf.device("/gpu:%d"%(i,)):
      if i == 0:
        x = tf.zeros((), dtype=np.float32)
        params = tf.Variable(1, dtype=np.float32, name="params")
        cost = tf.square(tf.reduce_sum(params-x), name="cost")
        [grad] = tf.gradients([cost], [params])
        templ = GraphTemplate([params, x, cost], [grad])
        inputs.append(x)
        costs.append(cost)
        grads.append(grad)
        params_list.append(params)
        train_ops.append(params.assign_sub(0.5*grad))
      else:
        x = tf.zeros((), dtype=np.float32)
        params = tf.Variable(1, dtype=np.float32, name="params")
        cost = tf.square(tf.reduce_sum(params-x), name="cost")
        [grad] = templ.apply([params, x, cost])
        inputs.append(x)
        costs.append(cost)
        grads.append(grad)
        params_list.append(params)
        train_ops.append(params.assign_sub(0.5*grad))
        
  # make train op
  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  check_equal(sess.run([costs, params_list, grads]),
              [[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
  sess.run(train_ops[0])

  check_equal(sess.run([costs, params_list, grads]),
              [[0.0, 1.0], [0.0, 1.0], [0.0, 2.0]])

  sess.run(train_ops[1])

  check_equal(sess.run([costs, params_list, grads]),
              [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])


def run_all_tests(module):
  all_functions = inspect.getmembers(module, inspect.isfunction)
  for name,func in all_functions:
    if name.endswith("_test"):
      print("Testing "+name)
      tf.reset_default_graph()
      func()
  print(module.__name__+" tests passed.")

if __name__=='__main__':
  #  run_all_tests(sys.modules[__name__])
  capture_ops_test()
