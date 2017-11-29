# simple example of launching tensorflow job

import time
import tensorflow as tf

flags = tf.flags
flags.DEFINE_string("role", "launcher", "either launcher or worker")
flags.DEFINE_integer("data_mb", 128, "size of vector in MBs")
flags.DEFINE_integer("iters_per_step", 10, "number of additions per step")
flags.DEFINE_string("cluster", "aws", "where to run (aws or local)")
FLAGS = flags.FLAGS

  
def main():
  if FLAGS.role == "launcher":
    launcher()
  elif FLAGS.role == "worker":
    worker()
  else:
    assert False, "Unknown role "+FLAGS.role


def launcher(do_local=False):
  if FLAGS.cluster == 'local':
    import tmux
    job = tmux.tf_job('myjob', 1)
  elif FLAGS.cluster == 'aws':
    import aws
    job = aws.tf_job('myjob', 1)
  else:
    assert False, "Unknown cluster "+FLAGS.cluster

  task = job.tasks[0]
  task.upload(__file__)   # copies current script onto machine
  setup_cmd =  ("source ~/.bashrc && export PATH=~/anaconda3/bin:$PATH && "
                "source activate tf")
  task.run("%s && python %s --role=worker" % (setup_cmd, __file__,))
  
  print("To see the output: tail -f %s" %(task.last_stdout))
  print("To interact with the task, do "+task.connect_instructions)
 

def worker():
  """Worker script that runs on AWS machine. Adds vectors of ones forever,
  prints MB/s."""
  
  def session_config():
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)
    config = tf.ConfigProto(
      graph_options=tf.GraphOptions(optimizer_options=optimizer_options))
    config.operation_timeout_in_ms = 10*1000  # abort after 10 seconds
    return config

  params_size = 250*1000*FLAGS.data_mb # 1MB is 250k floats
  dtype=tf.float32
  val = tf.ones((), dtype=dtype)
  vals = tf.fill([params_size], val)
  params = tf.Variable(vals)
  update = params.assign_add(vals)
  
  sess = tf.Session(config=session_config())
  sess.run(params.initializer)
  
  while True:
    start_time = time.perf_counter()
    for i in range(FLAGS.iters_per_step):
      sess.run(update.op)

    elapsed_time = time.perf_counter() - start_time
    rate = float(FLAGS.iters_per_step)*FLAGS.data_mb/elapsed_time
    print('%.2f MB/s'%(rate,))    

    
if __name__=='__main__':
  main()
