#!/usr/bin/env python
# Parse nvidia-smi for pids and kill all GPU users
# Tested on nvidia-smi 370.23
import os, re, sys, subprocess
import pwd

from collections import defaultdict

def tokenize(cmd):
  if isinstance(cmd, list):
    return cmd
  if isinstance(cmd, bytes):
    cmd = cmd.decode("ascii")
  if isinstance(cmd, str):
    cmd = cmd.split(None)
  return cmd


def run_command(cmd):
  """Run command, return output as string."""

  output = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            shell=True).communicate()[0]
  return output.decode("ascii")


def run_shell(cmd):
  """Runs shell command, returns list of outputted lines
  with newlines stripped."""

  cmd = tokenize(cmd)
  p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
  (stdout, stderr) = p.communicate()
  stdout = stdout.decode("ascii")  # turn into string to make Python3 happy
  lines = stdout.split('\n')
  stripped_lines = []
  for l in lines:
    stripped_line = l.strip()
    if l:
      stripped_lines.append(stripped_line)
  return stripped_lines


def run_shell_background(cmd_orig):
  """Runs shell command in background, returns pid."""

  cmd = tokenize(cmd_orig)
  p = subprocess.Popen(cmd, close_fds=True)
  print("[%d] %s " % (p.pid, cmd_orig))


def get_pid_gpu_map():
  """Returns map of GPU id to memory allocated on that GPU."""

  output = run_command("nvidia-smi")
  gpu_output = output[output.find("GPU Memory"):]
  # lines of the form
  # |    0      8734    C   python                             11705MiB |
  regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ]"
                     "(?P<gpu_memory>\d+)MiB")
  rows = gpu_output.split("\n")
  pids = []
  pid_gpu_map = defaultdict(list)
  for row in gpu_output.split("\n"):
    m = regex.search(row)
    if not m:
      continue
    pid = int(m.group("pid"))
    gpu_id = int(m.group("gpu_id"))
    print("pid %s using gpu %s"%(pid, gpu_id))
    pid_gpu_map[pid].append(gpu_id)
  return pid_gpu_map

def kill_pids(pids_to_kill):
  pids = []
  for pid_to_kill in pids_to_kill:
    pid = run_shell_background("sudo kill -9 "+str(pid_to_kill))
    pids.append(pid)
  return pids


def owner(pid):
  '''Return username of UID of process pid'''
  UID = 1
  EUID = 2
  for ln in open('/proc/%d/status' % pid):
    if ln.startswith('Uid:'):
      uid = int(ln.split()[UID])
      return pwd.getpwuid(uid).pw_name
          
if __name__ == '__main__':
  pid_gpu_map = get_pid_gpu_map()
  print("%10s %10s %s" %("pid", "username", "gpu"))
  for pid in pid_gpu_map:
    print("%10s %10s %s" %(pid, owner(pid), pid_gpu_map[pid]))
  answer = input("kill these? (Y/n) ")
  if not answer:
    answer = "y"
  if answer.lower() == "y":
    pids = kill_pids(pid_gpu_map.keys())
else:
    print("Didn't get y, doing nothing")
