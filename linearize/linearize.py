# utility to force a memory-efficient execution order on TensorFlow graph

import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
import toposort

def run_after(a, b):
   """Force operation a to run after b."""
   
   ge.reroute.add_control_inputs(a, [b])
   

# computation flows from parents to children

def parents(op):
  return set(input.op for input in op.inputs)
  
def children(op):
  return set(op for out in op.outputs for op in out.consumers())

def get_graph():
  """Creates dictionary {node: {child1, child2, ..},..} for current
  TensorFlow graph."""
  
  ops = tf.get_default_graph().get_operations()
  return {op: children(op) for op in ops}


def print_tf_graph(graph):
  """Prints tensorflow graph in dictionary form."""
  for node in graph:
    for child in graph[node]:
      print("%s -> %s" % (node.name, child.name))
    

def memsorted(nodes):
  """Sort nodes by estimated memory usage."""

  def node_memory(node, default_memory=1): return default_memory
  def node_name(node): return node.name
  
  def subtree_memory(node):
    return node_memory(node) + sum(node_memory(parent) for parent in parents(node))
  
  # sort by estimated memory, break ties alphabetically
  nodes = sorted(nodes, key=subtree_memory)
  nodes = sorted(nodes, key=node_name)
  return nodes
  
def linearize():
  """Add control dependencies to create a single valid execution order."""
  
  graph = get_graph()

  # find terminal nodes
  active = []
  for node in graph:
    if not graph[node]:  # no children
      active.append(node)

  # todo: sort first active set
  last_node = None
  for a in memsorted(active):
    print("Executing ", a.name)
    if last_node:
      run_after(last_node, a)
    last_node = a
    
  count = {node: len(graph[node]) for node in graph}
  # while true
  while active:
    new_active = []
    for node in active:  # also sort active set?
      for parent in memsorted(parents(node)):
        assert count[parent]>0
        count[parent]-=1
        if count[parent] == 0:
          new_active.append(parent)
          print("Executing ", parent.name)
          if last_node:
            run_after(last_node, parent)
          last_node = parent
            
    active = new_active

