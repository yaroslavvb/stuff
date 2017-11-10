#import exceptions
import functools
import logging
import os
import paramiko
import numpy
import sys
import threading
import time

def ExtractErrorToConsole(line):
  """Prints errors found in output to console
  
  Add checks to ensure any errors or info from the call of interest are 
  shown in the console to improve speed of identifying issues, e.g. socket 
  already used on non worker_0.  

  """
  # tf_cnn_bench error lines start with E
  if line.find('E') == 0:
    print(line.rstrip('\n'))
    return

  # Tensorflow Errors often look liked 'E tensorflow'
  if line.find('E tensorflow') != -1:
    print(line.rstrip('\n'))
    return

  # A little noisy but useful
  # Print number of devices found.  AWS often has
  # a busted GPU
  if line.find('DMA: ') != -1:
    print(line.rstrip('\n'))
    return

def ExtractToStdout(line):
  print(line.rstrip('\n'))


def ExtractImagePerSecond(line):
  if 'images/sec:' in line:
    print(line.rstrip('\n'))


def ExecuteCommandAndWait(ssh_client, command, print_error=True, ok_exit_status=[0]):
  _, stdout, stderr = ssh_client.exec_command(command, get_pty=True)
  exit_status = stdout.channel.recv_exit_status()
  if exit_status in ok_exit_status:
    return True
  else:
    if print_error:
      print('Error({}) executing command:{}'.format(exit_status, command))
      print(stdout.read())
    return False


def ExecuteCommandAndReturnStdout(ssh_client, command):
  _, stdout, stderr = ssh_client.exec_command(command, get_pty=True)
  exit_status = stdout.channel.recv_exit_status()
  return stdout.read()


def _StreamOutputToFile(fd, file, line_extractor, command=None):
  """Stream output to local file print select content to console

  Streams output to a local file and if a line_extractor is passed
  uses it to determine which data is printed to the local console.

  """
  def func(fd, file, line_extractor):
    with open(file, 'ab+') as f:
      if command:
        line = command + '\n'
        f.write(line.encode('utf-8'))
      try:
        for line in iter(lambda: fd.readline(2048), ''):
          f.write(line.encode('utf-8', errors='ignore'))
          f.flush()
          if line_extractor:
            line_extractor(line)
      except UnicodeDecodeError as err:
        print('UnicodeDecodeError parsing stdout/stderr, bug in paramiko:{}'
              .format(err))
  t = threading.Thread(target=func, args=(fd, file, line_extractor))
  t.start()
  return t


def ExecuteCommandAndStreamOutput(ssh_client,
                                  command,
                                  stdout_file=None,
                                  stderr_file=None,
                                  line_extractor=None,
                                  print_error=False,
                                  ok_exit_status=[0]):
  """Executes command in ssh_client.  Blocking call.


  Args:
    ssh_client: ssh client setup to connect to the server to run the tests on
    command: command to run in the ssh_client
    stdout_file: local file to write standard output of the command to
    stderr_file: local file to write standard error of the command to
    line_extractor: method to call on each line to determine if the line
    should be printed to the local console.
    print_error: True to print output if there is an error
    ok_exit_status: List of status codes that are not errors, defaults to '0'

  """
  _, stdout, stderr = ssh_client.exec_command(command, get_pty=True)
  if stdout_file:
    t1 = _StreamOutputToFile(stdout, stdout_file, line_extractor, command=command)
  if stderr_file:
    t2 = _StreamOutputToFile(stderr, stderr_file, line_extractor)
  if stdout_file:
    t1.join()
  if stderr_file:
    t2.join()
  exit_status = stdout.channel.recv_exit_status()
  if exit_status in ok_exit_status:
    return True
  else:
    if print_error:
      print('Command execution failed! Check log. Exit Status({}):{}'.format(exit_status, command))
    return False


def ExecuteCommandInThread(ssh_client,
                           command,
                           stdout_file=None,
                           stderr_file=None,
                           line_extractor=None,
                           print_error=False):
  """Returns a thread that executes the given command.  Non-Blocking call.

  

  Args:
    ssh_client: ssh client setup to connect to the server to run the tests on
    command: command to run in the ssh_client
    stdout_file: local file to write standard output of the command to
    stderr_file: local file to write standard error of the command to
    line_extractor: method to call on each line to determine if the line
    should be printed to the local console.
    print_error: True to print output if there is an error, e.g. non-'0' exit code.

  returns a thread that executes the given command

  """
  t = threading.Thread(
      target=ExecuteCommandAndStreamOutput,
      args=(ssh_client, command, stdout_file, stderr_file, line_extractor,
            print_error))
  t.start()
  return t


def SshToHost(hostname,
              retry=10,
              ssh_key=os.path.join(os.environ['HOME'], 'd/yaroslav.pem'),
              password=None,
              username='ubuntu'):

  """Create ssh connection to host

  Creates and returns and ssh connection to the host passed in.  

  Args:
    hostname: host name or ip address of the system to connect to.
    retry: number of time to retry.
    ssh_key: full path to the ssk hey to use to connect.
    username: username to connect with.

  returns SSH client connected to host.

  """

  # If logging is needed
  # paramiko.util.log_to_file("/tmp/paramiko.log")
  k = None
  if ssh_key: 
    k = paramiko.RSAKey.from_private_key_file(ssh_key)
  
  ssh_client = paramiko.SSHClient()
  ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

  counter = retry
  while counter > 0:
    try:
      if password: 
        ssh_client.connect(hostname=hostname, username=username, password=password)
      else:
        ssh_client.connect(hostname=hostname, username=username, pkey=k)
      break
    except Exception as e:
      counter = counter - 1
      print('Exception connecting to host via ssh (could be a timeout):'.format(e))
      if counter == 0:
        print('Got impatient with retrying ssh to host. Time to give up.')
        return None

  return ssh_client
