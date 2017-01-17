# script for getting cpu profile of queue runners
# 
# sudo apt-get install google-perftools
# LD_PRELOAD has to be set in a forked script, otherwise shell will
# overwrite the profile file

import os, sys, subprocess

my_env = os.environ.copy()
my_env["LD_PRELOAD"]="/usr/lib/libtcmalloc_and_profiler.so.4"
my_env["CPUPROFILE"]="/tmp/profile-yue/profile"

args = ["python", "benchmark-batch-noqueuerunners.py"]
proc = subprocess.Popen(args, stderr=subprocess.STDOUT, env=my_env)
print("Done")

