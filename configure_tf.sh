#!/usr/bin/expect -d
# Helper script that uses expect to automatically go through all configure
# steps using the defaults for all options except
# XLA: y
# CUDA: y
# compute capability: 3.5,5.0,6.0,6.1
spawn ./configure
expect "Please specify the location of python*"
send "\r"
expect "Please specify optimization flags to use during compilation when bazel option*"
send "\r"
expect "Do you wish to use jemalloc*"
send "\r"
expect "Do you wish to build TensorFlow with Google Cloud Platform*"
send "\r"
expect "Do you wish to build TensorFlow with Hadoop File System support*"
send "\r"
expect "Do you wish to build TensorFlow with the XLA*"
send "y\r"
expect "Please input the desired Python library*"
send "\r"
expect "Do you wish to build TensorFlow with OpenCL*"
send "\r"
expect "Do you wish to build TensorFlow with CUDA*"
send "y\r"
expect "Please specify which gcc should*"
send "\r"
expect "Please specify the CUDA SDK version you want to use*"
send "\r"
expect "Please specify the location where CUDA  toolkit*"
send "\r"
expect "Please specify the Cudnn version*"
send "\r"
expect "Please specify the location where cuDNN"
send "\r"
expect "lease specify a list of comma-separated Cuda compute"
send "3.5,5.2,6.0,6.1\r"
set timeout 120
expect eof
