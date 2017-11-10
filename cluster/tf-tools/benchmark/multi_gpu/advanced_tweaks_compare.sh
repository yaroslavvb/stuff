# Showing NCHW vs NHWC, NCCL and paramater server GPU vs CPU
_NUM_GPUS=1,2,8
LOG_FOLDER=advanced_tests

# PS GPU vs. CPU NHWC
./test_runner.sh --model resnet50 --num_batches 100 --batch_size 32 --gpus ${_NUM_GPUS} --ps_server gpu --variable_update send_recv --data_format NHWC --log_folder_prefix ${LOG_FOLDER} --framework tensorflow
./test_runner.sh --model resnet50 --num_batches 100 --batch_size 32 --gpus ${_NUM_GPUS} --ps_server cpu --variable_update send_recv --data_format NHWC --log_folder_prefix ${LOG_FOLDER} --framework tensorflow

# NCHW vs NHWC  (GPU PS)
./test_runner.sh --model resnet50 --num_batches 100 --batch_size 32 --gpus ${_NUM_GPUS} --ps_server gpu --data_format NCHW --variable_update send_recv --log_folder_prefix ${LOG_FOLDER} --framework tensorflow
./test_runner.sh --model resnet50 --num_batches 100 --batch_size 32 --gpus ${_NUM_GPUS} --ps_server gpu --data_format NHWC --variable_update send_recv --log_folder_prefix ${LOG_FOLDER} --framework tensorflow

# Add NCCL to NCHW (GPU PS)
./test_runner.sh --model resnet50 --num_batches 100 --batch_size 32 --gpus ${_NUM_GPUS} --ps_server gpu --data_format NCHW --variable_update replicated --log_folder_prefix ${LOG_FOLDER} --framework tensorflow


