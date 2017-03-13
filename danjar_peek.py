import tensorflow as tf
from tensorflow.python.client import timeline


class Queue(tf.FIFOQueue):

  def __init__(self, capacity):
    s = ()
    d = tf.int32
    super().__init__(capacity - 1, [d], [s])
    self._first = tf.get_variable(name="var1",
                                  initializer=tf.ones_initializer(),
                                  shape=s, dtype=d, use_resource=False)
    self._size = tf.get_variable(name="size", shape=(),
                                 initializer=tf.zeros_initializer(),
                                 dtype=tf.int32, use_resource=False)
 
  def peek(self):
    return self._first.read_value()

  def enqueue(self, element):
    super_ = super()
    def first():
      assigns = [self._first.assign(element)]
      with tf.control_dependencies(assigns):
        return tf.constant(0)
      
    def other():
      with tf.control_dependencies([super_.enqueue(element)]):
        return tf.constant(0)
      
    with tf.control_dependencies([self._size.assign_add(1)]):
      dummy = tf.cond(tf.equal(self._size, 0), first, other)
      return tf.identity(dummy)


queue = Queue(10)
queue_peek = queue.peek()
print("Peek op is "+str(queue_peek))

queue_init = queue.enqueue(tf.constant(-2))


print(tf.get_default_graph().as_graph_def())
for i in range(20):
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  sess.run(queue_init)
  print("queue size", sess.run(queue.size()))
  sess.run(queue.close())

#  print("Printing queue")
#  while True:
#    print(sess.run(queue.dequeue()))

  run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_options.output_partition_graphs = True
  run_metadata = tf.RunMetadata()
  #import pdb; pdb.set_trace()
  # queue_peek, 
  result = sess.run(queue_peek, run_metadata=run_metadata,
                    options=run_options)

  tl = timeline.Timeline(run_metadata.step_stats)
  ctf = tl.generate_chrome_trace_format()
  with open('timeline-%d.json'%(i,), 'w') as f:
    f.write(ctf)
  with open('stepstats-%d.json'%(i,), 'w') as f:
    f.write(str(run_metadata))

  print(result, end=' ')
  
# Expected: 1 1 1 1 1 1 1 1 1 1
# Actual: 0 1 0 0 1 1 0 0 0 1
