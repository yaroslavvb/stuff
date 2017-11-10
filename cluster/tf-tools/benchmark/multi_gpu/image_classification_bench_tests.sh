# Runs tests for an 8 GPU server
_NUM_GPUS=1,2,4,8
# Inception v3
./test_runner.sh --model inception3 --num_batches 100 --batch_size 32 --gpus ${_NUM_GPUS} --framework tensorflow
./test_runner.sh --model inception3 --num_batches 4 --batch_size 32 --gpus ${_NUM_GPUS} --framework mxnet


# Resnet-50
./test_runner.sh --model resnet50 --num_batches 100 --batch_size 32 --gpus ${_NUM_GPUS} --framework tensorflow
./test_runner.sh --model resnet50 --num_batches 4 --batch_size 32 --gpus ${_NUM_GPUS} --framework mxnet


# Resnet-152
./test_runner.sh --model resnet152 --num_batches 100 --batch_size 32 --gpus ${_NUM_GPUS} --framework tensorflow
./test_runner.sh --model resnet152 --num_batches 4 --batch_size 32 --gpus ${_NUM_GPUS} --framework mxnet


# AlexNet (OWT)
# AlexNet script is broken on MXNet.
#./test_runner.sh --model alexnet --num_batches 4 --batch_size 512 --gpus ${_NUM_GPUS} --framework mxnet
./test_runner.sh --model alexnet --num_batches 100 --batch_size 512 --gpus ${_NUM_GPUS} --framework tensorflow
./test_runner.sh --model alexnet --num_batches 100 --batch_size 128 --gpus ${_NUM_GPUS} --framework tensorflow
