import linearize

import os, sys, time
import inspect
import numpy as np
import tensorflow as tf
import pdb
import math
import toposort

from tensorflow.python.ops import gen_random_ops

def create_session():
  config = tf.ConfigProto(log_device_placement=False, graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
  return tf.InteractiveSession(config=config)

def setup_env():
  """Sets up test enviornment."""
  
  # download memory_util if needed
  memory_util_url = "https://raw.githubusercontent.com/yaroslavvb/memory_util/master/memory_util.py"
  if os.path.exists('memory_util.py'):
    size = len(open('memory_util.py').read())
  else:
    size = 0
    
  if size != 13636:
    print("Size changed or 0, redownloading memory_util.py")
    import urllib.request
    response = urllib.request.urlopen(memory_util_url)
    open("memory_util.py", "wb").write(response.read())

    
def make_caterpillar_graph(length=5, node_mbs=1):
  """Length is number of concats."""
  
  n = node_mbs * 250000
  n2 = int(math.sqrt(n))
  dtype = tf.float32
    
  def make_leaf(i):
    name = "leaf"+str(i)
    val = gen_random_ops._random_uniform((n2, n2), dtype, name=name)
    return val
   
  def make_merge(a, b, i):
    name = "merge"+str(i)
    merge_node = tf.matmul(a, b, name=name)
    #    nonlinear_node = tf.tanh(merge_node, name="tanh"+str(i))
    #nonlinear_node = tf.identity(merge_node, name="tanh"+str(i))
    return merge_node

  leaf0 = make_leaf(0)
  node0 = tf.identity(leaf0, name="merge0")
  node = node0
  nodes = [node]
  
  for i in range(1, length+1):
    leaf = make_leaf(i)
    node = make_merge(node, leaf, i)
    nodes.append(node)
  return nodes

def test_print():
  """Should print:
  leaf1 -> merge1
  leaf0 -> merge0
  merge1 -> merge2
  merge0 -> merge1
  leaf2 -> merge2
  leaf0/shape -> leaf0
  leaf1/shape -> leaf1
  leaf2/shape -> leaf2
  """
  
  nodes = make_caterpillar_graph(length=2)
  linearize.print_tf_graph(linearize.get_graph())
  

def test_toposort():
  nodes = make_caterpillar_graph(length=2)
  graph = linearize.get_graph()
  print(list(toposort.toposort(graph)))


def test_linearize():
  nodes = make_caterpillar_graph(5)
  linearize.linearize()

  sess = create_session()

  import memory_util
  memory_util.vlog(1)
  with memory_util.capture_stderr() as stderr:
    sess.run(nodes[-1].op)
  memory_util.print_memory_timeline(stderr, ignore_less_than_bytes=1000)

if __name__=='__main__':
  setup_env()
  import memory_util
  memory_util.vlog(1)
  
  #  sess = create_session()
  #nodes = make_caterpillar_graph()
  #  test_print()
  #  linearize.print_tf_graph(linearize.get_graph())
  #  print(tf.get_default_graph().as_graph_def())
  #  test_toposort()
  test_linearize()
  sys.exit()
  #  with memory_util.capture_stderr() as stderr:
  #    print(sess.run(nodes[-1][0,0]))
  print(len(stderr.getvalue()))
  memory_util.print_memory_timeline(stderr, ignore_less_than_bytes=1000)
