from scipy import linalg  # for svd
import urllib.request
import numpy as np

url="https://storage.googleapis.com/tensorflow-community-wheels/svd_in"
response = urllib.request.urlopen(url)
body = response.read()
print("Read %d bytes"%(len(body),))
assert len(body) == 15366400
open("svd_in", "wb").write(body)

dtype = np.float32
matrix0 = np.genfromtxt('svd_in',
                        delimiter= ",").astype(dtype)
assert matrix0.shape == (784, 784)
u, s, v = linalg.svd(matrix0)
print("matrix0 any NaNs: %s"% (np.isnan(matrix0).any(),))
print("u had NaNs: %s"% (np.isnan(u).any(),))
