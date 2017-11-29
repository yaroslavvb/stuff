# Performance

Reproducing 64-GPU ImageNet performance benchmark on AWS

Run this:

```
python launch.py --num_workers=8 --num_ps=4
```

That will launch 8 gradient workers on `p2.8xlarge` instances, and 4 parameter server workers on `c5.large` instances, and stream worker output locally.

For different sizes you should see this performance:


<img src=https://i.stack.imgur.com/CupI1.png>

# Detailed instructions
## Setup

First create the proper environment that can run this benchmark. This means installing TensorFlow and CUDA if necessary.

For this, launch an instance on AWS, SSH into it, setup all the proper packages, and then save this instance image as an AMI to use in your experiments.

Starting from Amazon Ubuntu 16.04 official AMI, might do this

```
sudo apt install -y wget python-pip git python-dev jq
sudo apt install -y python3-pip python3-dev jq
sudo apt install -y
sudo apt install python3
sudo apt install -y jq

curl -O https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh
bash ./Anaconda3-4.4.0-Linux-x86_64.sh -b -p ~/anaconda3

echo 'export PATH="/home/ubuntu/anaconda3/bin:$PATH"' >> ~/.bashrc
echo '# comment' >> ~/.bashrc
source ~/.bashrc
yes | conda update conda

yes | conda create -n tf python=3.5 anaconda
source activate tf
pip install tf-nightly-gpu --upgrade --upgrade-strategy=only-if-needed
yes | conda install pytorch torchvision cuda80 -c soumith
conda install -y -c conda-forge opencv

cudnn5="libcudnn5_5.1.10-1_cuda8.0_amd64.deb"
cudnn6="libcudnn6_6.0.21-1_cuda8.0_amd64.deb"
cuda_base="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/"
cuda_ver="cuda-repo-ubuntu1604_8.0.61-1_amd64.deb"

wget $cuda_base/$cuda_ver
sudo dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt -y update
sudo apt install -y "cuda-8.0"

code_url_base="https://storage.googleapis.com/studio-ed756.appspot.com/src"
wget $code_url_base/$cudnn5
wget $code_url_base/$cudnn6
sudo dpkg -i $cudnn5
sudo dpkg -i $cudnn6

echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo '# comment' >> ~/.bashrc
source ~/.bashrc
```

Now git clone TensorFlow benchmarks directory under `git0` directory
```
mkdir ~/git0
cd ~/git0
git clone https://github.com/tensorflow/benchmarks
```

Finally create an image from this instance,
<img src=https://i.stack.imgur.com/3iRWY.png>

After image is created, save it's AMI id into env var, ie

```
export AMI = 'ami-9ddb0xxx'
```


Now pick a security group that has SSH ports open (either pick an existing one from Security Groups tab on your console left sidebar, or create a new one)

Save name of the group into env var. It is under "Group Name" column in the console
<img src=https://i.stack.imgur.com/vqUTL.png>

security-group.png

Save the name of this group into environment variable
```
export SECURITY_GROUP=open
```


Now to configure ssh keys, go through the process of SSHing into some AWS instance. As a part of this, you will create a key with corresponding `.pem` file
Locate this file and save it's path to SSH_KEY_PATH env var

```
export SSH_KEY_PATH=~/home.pem
```

Finally, save the name of the keypair corresponding to this file into KEY_NAME env variable (it's the "Key pair name" under Network & Security/Key Pairs)

```
export KEY_NAME=mykey
```

## Running experiment

Now to launch experiment with 1 gradient worker and 1 parameter server

```
pip install -r requirements.txt
python launch.py
```

After instances launch, you should see something like this printed on your console for each task launched (1 for ps, and 1 for gradient worker)

```
...
To see the output of 0: tail -f /temp/tasklogs/cnn1p3/worker/0/884ABE537587D641.stdout
```
Execute this command in a different console and you'll see the results streaming in real time

```
...
2017-11-28 21:08:37.690888: I tensorflow/core/distributed_runtime/master_session.cc:1008] Start master session 158197ad72c6e80a with config: intra_op_parallelism_threads: 1 gpu_options { force_gpu_compatible: true } allow_soft_placement: true
Running warm up
Done warm up
Step	Img/sec	loss
1	images/sec: 373.4 +/- 0.0 (jitter = 0.0)	7.924
10	images/sec: 369.9 +/- 4.2 (jitter = 1.3)	7.888
20	images/sec: 369.6 +/- 3.0 (jitter = 1.5)	7.871
30	images/sec: 371.0 +/- 2.0 (jitter = 1.9)	7.884
40	images/sec: 371.5 +/- 1.5 (jitter = 1.7)	7.902
```


To launch multiple experiments, give them different run names. IE, to evaluate in parallel an experiment with 8 gradient workers, 4 ps workers, and use a specific instance type for each group, do this


```
python launch.py --run=cnn8 --num_workers=8 --num_ps=4
```


You may also want to customize instance types to get around your limits. Specify those as follows
```
python launch.py --run=cnn8 --num_workers=8 --num_ps=4 --worker_type=p2.8xlarge --ps_type=c5.2xlarge
```

To clean up your running experiments, use [terminate_instances.py](https://github.com/diux-dev/cluster/blob/master/terminate_instances.py) script. It's hardwired to only kill instances launched with a specific ssh key, so modify `LIMIT_TO_KEY` constant in the script to work for your instances.

## Troubleshooting

* Sometimes first launch will get stuck with the following error printed in worker console

```
tensorflow.python.framework.errors_impl.UnavailableError: Endpoint read failed
```

This is due one of the workers taking too long to spin up, and connection timing out in `tf_cnn_benchmarks.py`. The solution is to simply repeat the same launch command. It will reuse existing AWS instances, so the connection should succeed the second time.

* **Note for people porting to other systems**: "variable_mgr.py" in benchmarks is not compatible with Python3, hence the `launch.py` script overwrites it with the local `variable_mgr.py` version. If you don't use launch.py, make sure to copy this file manually.
