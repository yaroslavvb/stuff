import os, sys, subprocess

my_env = os.environ.copy()
my_env["LD_PRELOAD"]="/usr/lib/libtcmalloc_and_profiler.so.4"
my_env["CPUPROFILE"]="/tmp/profile-yue/profile"

args = ["python", "benchmark-batch-noqueuerunners.py"]
proc = subprocess.Popen(args, stderr=subprocess.STDOUT, env=my_env)
print("Done")

