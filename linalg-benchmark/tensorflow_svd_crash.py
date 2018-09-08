# matrix
# https://storage.cloud.google.com/tensorflow-community-wheels/svd_in
# Mathematica sanity check
# https://www.wolframcloud.com/objects/f16d71a7-cc47-4a3d-b686-da440670eed3

import tensorflow as tf
import numpy as np
import os


if not os.path.exists("svd_in"):
  import urllib.request
  url="https://storage.googleapis.com/tensorflow-community-wheels/svd_in"
  response = urllib.request.urlopen(url)
  body = response.read()
  print("Read %d bytes"%(len(body),))
  assert len(body) == 15366400
  open("svd_in", "wb").write(body)

  #  import requests
  # r = requests.get(url, auth=('usrname', 'password'), verify=False,stream=True)
  # r.raw.decode_content = True
  # with open("svd_in", 'wb') as f:
  #   shutil.copyfileobj(r.raw, f)

dtype = np.float32
matrix0 = np.genfromtxt('svd_in',
                        delimiter= ",").astype(dtype)
print(matrix0.shape)
assert matrix0.shape == (784, 784)
matrix = tf.placeholder(dtype)
sess = tf.InteractiveSession()
s0,u0,v0 = sess.run(tf.svd(matrix), feed_dict={matrix: matrix0})
print("u any NaNs: %s"% (np.isnan(u0).any(),))
print("u all NaNs: %s"% (np.isnan(u0).all(),))
print("matrix0 any NaNs: %s"% (np.isnan(matrix0).any(),))

# segfault bt
# #0  0x00007fffe320e121 in Eigen::BDCSVD<Eigen::Matrix<float, -1, -1, 1, -1, -1> >::perturbCol0(Eigen::Ref<Eigen::Array<float, -1, 1, 0, -1, 1>, 0, Eigen::InnerStride<1> > const&, Eigen::Ref<Eigen::Array<float, -1, 1, 0, -1, 1>, 0, Eigen::InnerStride<1> > const&, Eigen::Ref<Eigen::Array<long, 1, -1, 1, 1, -1>, 0, Eigen::InnerStride<1> > const&, Eigen::Matrix<float, -1, 1, 0, -1, 1> const&, Eigen::Ref<Eigen::Array<float, -1, 1, 0, -1, 1>, 0, Eigen::InnerStride<1> > const&, Eigen::Ref<Eigen::Array<float, -1, 1, 0, -1, 1>, 0, Eigen::InnerStride<1> > const&, Eigen::Ref<Eigen::Array<float, -1, 1, 0, -1, 1>, 0, Eigen::InnerStride<1> >) ()
#    from /home/yaroslav/.conda/envs/whitening/lib/python3.5/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
# #1  0x00007fffe320fa81 in Eigen::BDCSVD<Eigen::Matrix<float, -1, -1, 1, -1, -1> >::computeSVDofM(long, long, Eigen::Matrix<float, -1, -1, 0, -1, -1>&, Eigen::Matrix<float, -1, 1, 0, -1, 1>&, Eigen::Matrix<float, -1, -1, 0, -1, -1>&) ()
#    from /home/yaroslav/.conda/envs/whitening/lib/python3.5/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
# #2  0x00007fffe321e21c in Eigen::BDCSVD<Eigen::Matrix<float, -1, -1, 1, -1, -1> >::divide(long, long, long, long, long) ()
#    from /home/yaroslav/.conda/envs/whitening/lib/python3.5/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
# #3  0x00007fffe321dbb8 in Eigen::BDCSVD<Eigen::Matrix<float, -1, -1, 1, -1, -1> >::divide(long, long, long, long, long) ()
#    from /home/yaroslav/.conda/envs/whitening/lib/python3.5/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
# #4  0x00007fffe32220bd in Eigen::BDCSVD<Eigen::Matrix<float, -1, -1, 1, -1, -1> >::compute(Eigen::Matrix<float, -1, -1, 1, -1, -1> const&, unsigned int) ()
#    from /home/yaroslav/.conda/envs/whitening/lib/python3.5/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
# #5  0x00007fffe32227a1 in tensorflow::SvdOp<float>::ComputeMatrix(tensorflow::OpKernelContext*, tensorflow::gtl::InlinedVector<Eigen::Map<Eigen::Matrix<float, -1, -1, 1, -1, -1> const, 0, Eigen::Stride<0, 0> >, 4> const&, tensorflow::gtl::InlinedVector<Eigen::Map<Eigen::Matrix<float, -1, -1, 1, -1, -1>, 0, Eigen::Stride<0, 0> >, 4>*) ()                                                                 from /home/yaroslav/.conda/envs/whitening/lib/python3.5/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so                                       #6  0x00007fffe3228c75 in tensorflow::LinearAlgebraOp<float>::ComputeTensorSlice(tensorflow::OpKernelContext*, long long, tensorflow::gtl::InlinedVector<tensorflow::Tensor const*, 4> const&, tensorflow::gtl::InlinedVector<tensorflow::TensorShape, 4> const&, tensorflow::gtl::InlinedVector<tensorflow::Tensor*, 4> const&, tensorflow::gtl::InlinedVector<tensorflow::TensorShape, 4> const&) ()
#    from /home/yaroslav/.conda/envs/whitening/lib/python3.5/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so
   

# upload notes

# export fullname=svd_in
# export bucket=tensorflow-community-wheels
# cd ~/git0/whitening/mnist_autoencoder/data
# gsutil cp svd_in gs://$bucket
# gsutil acl set public-read gs://$bucket/$fullname
# echo https://storage.googleapis.com/tensorflow-community-wheels/$fullname
