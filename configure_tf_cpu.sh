#!/usr/bin/expect -d
# Helper script that uses expect to automatically go through all configure
# steps using the defaults for all options except
# XLA: y
# CUDA: n
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
send "\r"
set timeout 120
expect eof
