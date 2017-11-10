# Install for Amazon Linux TensorFlow + CUDA
This install was created on the latest Amazon Linux image as of 12-FEB-2017.
The end results is a system that can build and run TensorFlow.  To save space
create one machine to do builds following these instructions and then another 
AMI to run tensorflow without the source code, bazel and development kit items.
Most of the tools are necessary to run TensorFlow but I collected some of the
command from other locatios and thus some of the libs and tools installed may
not actually be needed.  I also install a few tools I use.

## Which AMI and size
Pick the most recent AWS Amazon Linux image.  I chose 20GB to be 100% sure 
everything would fit for a "build box".  I also plan to run other frameworks.
If building something to just run TensorFlow, 8GB is likely enough.  

## Is this perfect?
Not remotely.  I try to keep notes each time I do it and it gets better each
time.  I also sometimes forget to copy the commands back to this file.  If it
is not exact it will get you very close.  

## Fun Tip (to save money)
Start the instance on a m4 instance.  Skip the NVIDIA driver install. Everything
else will work.  Then save off as an AMI and launch as a p2 instance.  A few
notes:
*   m4 instances do not have avx2.  Cross compile will work but the binary
    cannot be tested.  Just compile without avx2 to test.
*   Compiling with CUDA works but will not run on m4, because it does not have
    an NVIDIA card.  
*   After moving to p2, make sure to install the NVIDIA driver
*   This might not save money, it does for me becuase I end up keeping an m4 
    instance as a buid box.  So I can build gcc 4.8.x.  

## Install
Some [hints](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/accelerated-computing-instances.html#optimize_gpu)
from Amazon.  I did not have good results turning off auto-boost.

```bash

#######################################################
# The basics, do this even if skipping other sections
######################################################
sudo yum update -y
sudo yum erase nvidia cuda

sudo yum install -y gcc kernel-devel-`uname -r`
sudo yum install python-numpy gcc-c++ git

Only needed for Bazel to build TF, skip if not buildig a build machine
sudo yum remove java-1.7.0-openjdk
sudo yum install java-1.8.0 java-1.8.0-openjdk-devel

*Setup pip (python package manager)*
sudo pip install --upgrade pip
# Make a symbolic link so sudo can see pip (cheap hack)
cd /usr/bin
sudo ln -s /usr/local/bin/pip

# Upgrade some python packages
pip install awscli --upgrade --user
sudo pip install wheel
sudo pip install numpy

############################################
# Installing NVIDIA with CUDA and cuDNN
#############################################
# Install NVIDIA Driver
# This driver.  I ended up installing 367.48 from the CUDA download.
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/352.99/NVIDIA-Linux-x86_64-352.99.run
sudo ./NVIDIA-Linux-x86_64-352.99.run
sudo reboot

# Install CUDA
# usr a newer version if desired
wget https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_linux-run
./cuda_8.0.44_linux-run --extract=/home/ec2-user/
sudo ./cuda-linux64-rel-8.0.44-21122537.run

# Install CuDNN
# Download CuDNN from NVIDIA (get the Linux package not deb packages)
# Need to sign up to down load from NVIDIA, I would share a link if I could
# this step is annoying
wget <Need to sign up to download from NVIDIA>
tar zxf cudnn-8.0-linux-x64-v5.1.tgz
cd cuda

# Copy files into CUDA directorys
sudo cp -P include/cudnn.h /usr/local/cuda-8.0/include/
sudo cp -P lib64/libcudnn* /usr/local/cuda-8.0/lib64/
sudo chmod a+r /usr/local/cuda-8.0/lib64/libcudnn*

# Setup Profile with CUDA environment variables
Add to ~.bash_profile
#CUDA Setup
export CUDA_HOME=/usr/local/cuda
export CUDA_ROOT=/usr/local/cuda
PATH=$PATH:$CUDA_ROOT/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64
source .bash_profile

############################################
# Installing TensorFlow from source
#############################################
# Download bazel get latest but the java install is fine

# set java home
export JAVA_HOME=/etc/alternatives/java_sdk
wget https://github.com/bazelbuild/bazel/releases/download/0.4.4/bazel-0.4.4-jdk7-installer-linux-x86_64.sh
# install to your $HOME/bin directory (find if you are not sharing)
./bazel-0.4.4-jdk7-installer-linux-x86_64.sh --user


# Build Tensorflow
cd ~
mkdir src
cd src
git clone https://github.com/tensorflow/tensorflow
cd tensorflow

./configure
Defaults with exception of
Y Cuda 8.0 
compute 3.7 (max for K80 assuming not using g.xx instances)

# Works for p2 AWS instances with broadwell CPUs.
# https://gcc.gnu.org/onlinedocs/gcc-4.8.5/gcc/i386-and-x86-64-Options.html#i386-and-x86-64-Options
# core-avx2 is the best option for hazwel with gcc 4.8.3
bazel build  --copt=-march=core-avx2 --config=cuda //tensorflow/tools/pip_package:build_pip_package
# or to build for the native machine
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

# Build the pip package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

# Install
cd /tmp/tensorflow_pkg
sudo pip install --upgrade --force-reinstall <tensorflow file>

# Do a quick Test and watch for any warnings, e.g. AVX FMA not compiled in
# Watch that the GPUs start up, assuming GPUs were desired
python src/tensorflow/tensorflow/examples/tutorials/mnist/mnist_softmax.py


################################
# Optional
################################

# ldcofig. There are some scenarios where .bash_profile does not seem to work
# I often set the following when running benchmark tests so I know
# cuda is always being picked up and reduces odd errors.

sudo vi /etc/ld.so.conf
/usr/local/cuda/lib64
/usr/local/cuda/extras/CUPTI/lib64
# Load new config
sudo ldconfig

# Verify config 
ldconfig -p


```



