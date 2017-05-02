import scipy.linalg as linalg
import numpy as np

def download_if_needed(fn,target_length=0,bucket="yaroslavvb_stuff"):
  import urllib.request
  url="https://storage.googleapis.com/%s/%s"%(bucket, fn)
  response = urllib.request.urlopen(url)
  body = response.read()
  print("Read %d bytes"%(len(body),))
  if target_length:
    assert len(body)==target_length
    
  open(fn, "wb").write(body)

fn='badsvd0'
download_if_needed(fn, 2458624)
target0 = np.fromfile(fn, np.float32).reshape(784,784)
u0, s0, vt0 = linalg.svd(target0)
print("Success")

# Upload notes:
# export fullname=badsvd0
# export bucket=yaroslavvb_stuff
# gsutil cp $fullname gs://$bucket
# gsutil acl set public-read gs://$bucket/$fullname
# echo https://storage.googleapis.com/tensorflow-community-wheels/$fullname
