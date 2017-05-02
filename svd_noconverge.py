import scipy.linalg as linalg
import numpy as np
import os
import sys
import ctypes
import numpy as np

def mklVersion():
    ver = np.zeros(199, dtype=np.uint8)
    mkl = ctypes.cdll.LoadLibrary("libmkl_rt.so")
    mkl.MKL_Get_Version_String(ver.ctypes.data_as(ctypes.c_char_p), 198)
    return ver[ver != 0].tostring()

# mklVersion()

def download_if_needed(fn,target_length=0,bucket="yaroslavvb_stuff"):
  import urllib.request
  url="https://storage.googleapis.com/%s/%s"%(bucket, fn)
  response = urllib.request.urlopen(url)
  body = response.read()
  print("Read %d bytes from $url"%(len(body),))
  if target_length:
    assert len(body)==target_length
    
  open(fn, "wb").write(body)

fn='badsvd0'
download_if_needed(fn, 2458624)
target0 = np.fromfile(fn, np.float32).reshape(784,784)

try:
  u0, s0, vt0 = linalg.svd(target0)
except Exception as e:
  print("SVD failure")
  print(e)
else:
  print("SVD success")

print("-"*80)
print("MKL version")
print(mklVersion())
print("-"*80)
print("Conda version")
os.system("conda list --explicit")
print("-"*80)
print("CPU version")
for l in open("/proc/cpuinfo").read().split('\n'):
  if 'model name' in l:
    print(l)
    break


# Upload notes:
# export fullname=badsvd0
# export bucket=yaroslavvb_stuff
# gsutil cp $fullname gs://$bucket
# gsutil acl set public-read gs://$bucket/$fullname
# echo https://storage.googleapis.com/$bucket/$fullname
