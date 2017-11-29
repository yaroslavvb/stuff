# simple example of launching tensorflow job

import aws
import os
import sys
import time
import tensorflow as tf
import boto3

flags = tf.flags
flags.DEFINE_string("role", "launcher", "either launcher or worker")
flags.DEFINE_integer("data_mb", 128, "size of vector in MBs")
flags.DEFINE_integer("iters_per_step", 10, "number of additions per step")
FLAGS = flags.FLAGS

module_path=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, module_path+'/tf-tools/benchmark/runner')
import cluster_aws as toby_aws


def test_new_job():
  name = "testjob"
  instances = toby_aws.LookupAwsInstances(instance_tag=name)
  assert not instances, "Instances already exist, kill them first"

  job = aws.tf_job(name, 2)
  instances = toby_aws.LookupAwsInstances(instance_tag=name)
  assert len(instances) == 2

def test_terminate_job():
  aws.terminate_job("testjob")


def test_reuse_job():
  name = "testjob"
  job = aws.tf_job(name, 2)

def test_send_file():
  name = "testjob"
  job = aws.tf_job(name, 4)
  job.wait_until_ready()
  task0 = job.tasks[0]
  secret_word = "testfile3"
  os.system("echo '%s' > upload_test.txt"%(secret_word,))
  task0.upload('upload_test.txt')
  stdout,stderr = task0.run_sync("cat upload_test.txt")
  print(stdout)    # => testfile2
  assert stdout.strip() == secret_word

def test_upload_directory():
  pass

def test_stream_output():
  name = "testjob"
  job = aws.tf_job(name, 4)
  job.wait_until_ready()
  task = job.tasks[0]
  task.run('cd Dropbox && ls') 
  time.sleep(0.5)  # async ... todo: expose thread and join instead of sleep?
  os.system('cat '+task.last_stdout)


def main():
  #  test_terminate_job()
  #  test_new_job()
  #  test_reuse_job()
  #  test_send_file()
  test_stream_output()
    
if __name__=='__main__':
  main()
