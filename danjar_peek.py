import tensorflow as tf
from tensorflow.python.client import timeline


class Queue(tf.FIFOQueue):

  def __init__(self, capacity, dtypes, shapes):
    super().__init__(capacity - 1, dtypes, shapes)
    self._first = [
        tf.Variable(tf.zeros(s, d), dtype=d, expected_shape=s)
        for s, d in zip(shapes, dtypes)]
    self._size = tf.Variable(0, tf.int32)

  def peek(self):
    with tf.control_dependencies([tf.assert_greater(self._size, 0)]):
      return tf.identity(self._first)

  def dequeue(self):
    super_ = super()
    def first():
      return tf.identity(self._first)
    def other():
      dequeue = super_.dequeue()
      dequeue = dequeue if isinstance(dequeue, tuple) else (dequeue,)
      assigns = [p.assign(s) for p, s in zip(self._first, dequeue)]
      with tf.control_dependencies(assigns):
        return tf.identity(self._first)
    element = tf.cond(tf.equal(self._size, 1), first, other)
    with tf.control_dependencies([self._size.assign_sub(1)]):
      return tf.identity(element)

  def enqueue(self, element):
    super_ = super()
    def first():
      assigns = [p.assign(e) for p, e in zip(self._first, element)]
      with tf.control_dependencies(assigns):
        return tf.constant(0)
    def other():
      with tf.control_dependencies([super_.enqueue(element)]):
        return tf.constant(0)
    dummy = tf.cond(tf.equal(self._size, 0), first, other)
    with tf.control_dependencies([self._size.assign_add(1)]):
      return tf.identity(dummy)


queue = Queue(10, [tf.int32], [()])
queue_peek = queue.peek()[0]
queue_init = queue.enqueue([tf.constant(1)])

for i in range(10):
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  sess.run(queue_init)

  run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_options.output_partition_graphs = True
  run_metadata = tf.RunMetadata()
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
