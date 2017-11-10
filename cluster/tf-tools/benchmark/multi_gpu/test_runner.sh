#!/bin/bash

# Set defaults to best performance for most scenarios and most used values
GPUS_PER_HOST=1
DATA_FORMAT=NCHW  #other option is NHWC (NCHW is cuDNN default format, will not work on CPU)
BATCH_SIZE=32
MODEL=inception3
DISPLAY_EVERY=10 #Display after every X steps.
NUM_BATCHES=100 # 4 or 5 is good for mxnet (their synthetic test does 40 batches per epoch)
MEM_TEST=false
VARIABLE_UPDATE=replicated #send-recv works on CPU and GPU.  For NVIDIA GPU use replicated, which uses nccl.
PS_SERVER=gpu # cpu often performs similar to gpu but GPU is recommended unless there is a memory constraint.
FRAMEWORK=tensorflow #m xnet is also supported

# Directories for github repos
SRC_ROOT="${HOME}/src" 
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# Only used for mxnet
NUM_LAYERS=50

#Run with Real data
SYNTHETIC_DATA=true
DATA_DIR=/usr/local/google/home/tobyboyd/imagenet

# Genearl Config
LOG_FOLDER_PREFIX=logs  # Useful for grouping logs

# Stats monitor
MONITOR_STATS=true

while [[ $# -gt -0 ]]; do
  key="$1"
  echo $key
  echo $2
  case $key in
      --gpus)
      GPUS_PER_HOST="$2" # which GPU configs to run , e.g.  1,2,4,8 (4 tests one for eaah number of GPUs)
      shift
      ;;
    --data_format)
      DATA_FORMAT="$2"  # Format of the data NHWC or NCHW (NVIDIA)
      shift
      ;;
    --num_batches)
      NUM_BATCHES="$2"  # Number of batches.  Note that mxnet uses epochs which is 40 per num_batches
      shift
      ;;
    --batch_size) # Size of the batch to run
      BATCH_SIZE="$2"
      shift
      ;;
    --display_every_step)
      DISPLAY_EVERY="$2" # how often to display batch timings, 10 is reasonable.
      shift
      ;;
    --variable_update)
      VARIABLE_UPDATE="$2"   # Set how to update variables
      shift
      ;;
    --mem_test)
      MEM_TEST="$2" # set to true to run a mem-test
      shift
      ;;
    --ps_server)
      PS_SERVER="$2"   # where to run the parameter server or where to place them cpu/gpu
      shift
      ;;
    --framework)
      FRAMEWORK="$2"   # framework to use, mxnet or tensorflow
      shift
      ;;
    --model)
      MODEL="$2"   # name of the model to run
      shift
      ;;
    --src_root)
      MODEL="$2"   # Folder where the github repos are found
      shift
      ;;
    --log_folder_prefix)
      LOG_FOLDER_PREFIX="$2"   # prefix for the log folder to help with organization
      shift
      ;;
    --data_dir)
      DATA_DIR="$2"   # prefix for the log folder to help with organization
      SYNTHETIC_DATA=false
      shift
      ;;
    --synthetic)
      SYNTHETIC_DATA="$2"   # prefix for the log folder to help with organization
      shift
      ;;
    --monitor_stats)
      MONITOR_STATS="$2"  # on by default set to false to turn off
      shift
      ;;
    *)
      echo "Unknown flag: $key"
      ;;
  esac
  shift # past argument or value
done

#Convert values for non-tensorflow platforms
if [ "${FRAMEWORK}" = "mxnet" ]; then
    #Cheap trick to align GPU index MXnet uses 0 base
    # TODO: make work for multiple GPU, eg. 1,2,4,8
   #GPUS=$((GPUS-1))

   IMAGE_SHAPE=not_set
   case $MODEL in
    inception3)
        IMAGE_SHAPE="3,299,299"
        MODEL=inception-v3
        ;;
    resnet50)
        IMAGE_SHAPE="3,224,224"
        MODEL=resnet
        ;;
    resnet152)
        IMAGE_SHAPE="3,224,224"
        MODEL=resnet
        NUM_LAYERS=152
        ;;
    alexnet)
        IMAGE_SHAPE="3,224,224"
        # MXNet has alenet and googlenet mixed up in their repo
        MODEL=alexnet
        ;;
    *)
        echo "Unknown mxnet model: $MODEL" # unknown option
        ;;
    esac 
fi

convertToMXNetGPU () 
{
    local mxnet_gpu=0
    case $1 in
    1)
      mxnet_gpu="0"
      ;;
    2)
      mxnet_gpu="0,1" 
      ;;
    4)
      mxnet_gpu="0,1,2,3"
      ;;
    8)
      mxnet_gpu="0,1,2,3,4,5,6,7" 
      ;;
    *)
      echo "Unknown GPU string $1"
      exit 1;
      ;;
    esac

    echo $mxnet_gpu
}

# Set log folder name and links to tools folders
LOG_FOLDER="${LOG_FOLDER_PREFIX}/${TIMESTAMP}_${FRAMEWORK}_${MODEL}_${BATCH_SIZE}"
MXNET_BENCHMARK_DIRECTORY="${SRC_ROOT}/mxnet/example/image-classification"
TF_BENCHMARK_DIRECTORY="${HOME}/tf_cnn_bench"


echo "##### Config #####"
echo "GPU configs to test: $GPUS"
echo "Parameter server: ${PS_SERVER}"
echo "data_format: $DATA_FORMAT"
echo "batch_size: $BATCH_SIZE"
echo "variable_update: $VARIABLE_UPDATE"
echo "batch_num: $NUM_BATCHES"
echo "mem_test: $MEM_TEST"
echo "framework: $FRAMEWORK"
echo "log folder: $LOG_FOLDER"
echo "src root: $SRC_ROOT"

#make log folder
echo "Make log folder:${LOG_FOLDER}"
mkdir -p $LOG_FOLDER

GPUS=${GPUS_PER_HOST//,/$'\n'}  # change the commas to white space

# PID for the stats monitor
STATS_MONITOR_PID=0
# Handle CTRL-C or other term signal, log the max number of GPUs that showed
# Errors, for now that means "HW Slowdown: Active"
function cleanup {
  # kill stats monitor incase it is running using all methods
  if [ "$STATS_MONITOR_PID" -gt "0" ]; then 
    kill $STATS_MONITOR_PID
    wait $STATS_MONITOR_PID
    pgrep "stats_monitor" | xargs kill >/dev/null 2>&1
    exit;
  fi
}

# catch being asked to end
trap cleanup SIGINT SIGTERM

# This is a little lame but double check stats_monitor was killed
pgrep "stats_monitor" | xargs kill >/dev/null 2>&1


# Execute the benchmark on each GPU config, e.g. 1, 2, 4 and then 8 GPUs.  
if [ "$MEM_TEST" = "false" ]; then
    for gpu in $GPUS; do
      if [ "${FRAMEWORK}" = "mxnet" ]; then
        gpu=$(convertToMXNetGPU $gpu)
        BENCH_EXEC="python ${MXNET_BENCHMARK_DIRECTORY}/train_imagenet.py --gpus ${gpu} --network ${MODEL} --batch-size ${BATCH_SIZE} \
--image-shape ${IMAGE_SHAPE} --num-epochs ${NUM_BATCHES} --kv-store device --benchmark 1 --disp-batches ${DISPLAY_EVERY} \
--num-layers ${NUM_LAYERS}"
      else  
        BENCH_EXEC="python ${TF_BENCHMARK_DIRECTORY}/tf_cnn_benchmarks.py --model=${MODEL} --batch_size=${BATCH_SIZE} --num_batches=${NUM_BATCHES} \
--num_gpus=${gpu} --data_format=${DATA_FORMAT} --display_every=${DISPLAY_EVERY} --weak_scaling=true \
--parameter_server=${PS_SERVER} --device=gpu --variable_update=${VARIABLE_UPDATE}"
        if [ "$SYNTHETIC_DATA" = "false" ]; then
          BENCH_EXEC="${BENCH_EXEC} --data_dir=${DATA_DIR} --nodistortions --data_name=imagenet"
        fi
      fi
      
      # run stats monitor (wathch for overheading cards)
      if [ "$MONITOR_STATS" = "true" ]; then
        echo "Start stats monitor..."
        ./stats_monitor.sh --log_full_path $LOG_FOLDER/${gpu}_full_log.txt --log_summary_full_path $LOG_FOLDER/${gpu}_log_summary.txt  &
        STATS_MONITOR_PID=$! 
      fi

      echo "Command to run: $BENCH_EXEC"
      $BENCH_EXEC 2>&1 | tee $LOG_FOLDER/${gpu}.txt
      
      if [ "$STATS_MONITOR_PID" -gt "0" ]; then  
        echo "Stats monitor: stopping"
        kill $STATS_MONITOR_PID
        echo "Wait for stats monitor to stop: ${STATS_MONITOR_PID}"
        wait $STATS_MONITOR_PID
        echo "Stats monitor: stopped"
      fi

      if [ "${FRAMEWORK}" = "mxnet" ]; then
        # average last 5 entries (not the most elegant string of commands)
        grep "samples/sec" $LOG_FOLDER/${gpu}.txt | awk '{print $5}' | tail -5 | \
awk '{ sum += $1; n++ } END { if (n > 0) print "Images/sec Last 5 Results:" sum / n }' 2>&1 | tee -a $LOG_FOLDER/${gpu}.txt 
      fi

    done

    echo "$BENCH_EXEC" >> $LOG_FOLDER/result.txt
    grep "^total images/sec" $LOG_FOLDER/*.txt 2>&1 | tee -a $LOG_FOLDER/result.txt
    # Gather up summary log messages into botto of results file
    cat $LOG_FOLDER/${gpu}_log_summary.txt >> $LOG_FOLDER/result.txt

fi


# Find where program fails due to memory errors by brute force
# Suggestion is to run 100 batches as a standard
if [ "$MEM_TEST" = "true" ]; then
    BATCH_ARRAY=(${BATCH_SIZE//,/$'\n'}) # Turn batch_size into an array expected LOW[0] HIGH[1]
    
    # convert $BATCH_SIZE into BATCH_SIZES, BATCH_SIZE expected to be a range, e.g. 200,300
    BIN_SEARCH_HIGH=${BATCH_ARRAY[1]}
    BIN_SEARCH_LOW=${BATCH_ARRAY[0]}
    BIN_SEARCH_RUN=$BIN_SEARCH_HIGH
    RESULT=0
    RESULT_TEXT="Never OOM, try a larger HIGH batch size"
    RESULT_NEVER_PASS=true
    GPU_CONVERSION=false
    #Distance moved will always be negative when moving down from OOM
    while [ $(($BIN_SEARCH_LOW + 1)) -lt $BIN_SEARCH_HIGH ]; do
        echo "Test batch size ${BIN_SEARCH_RUN}" 
        if [ "${FRAMEWORK}" = "mxnet" ]; then
          #do not convert on every loop
          if [ "${GPU_CONVERSION}" = "false" ]; then
            GPUS=$(convertToMXNetGPU $GPUS)
             GPU_CONVERSION=true
          fi

          BENCH_EXEC="python ${MXNET_BENCHMARK_DIRECTORY}/train_imagenet.py --gpus ${GPUS} --network ${MODEL} --batch-size ${BIN_SEARCH_RUN} \
--image-shape ${IMAGE_SHAPE} --num-epochs ${NUM_BATCHES} --kv-store device --benchmark 1 --disp-batches ${DISPLAY_EVERY} --num-layers ${NUM_LAYERS}"
          OOM_LOG_MSG="cudaMalloc failed: out of memory"
        else
          BENCH_EXEC="python ${TF_BENCHMARK_DIRECTORY}/tf_cnn_benchmarks.py --model=${MODEL} --batch_size=${BIN_SEARCH_RUN} --num_batches=${NUM_BATCHES} \
--num_gpus=${GPUS} --data_format=${DATA_FORMAT} --display_every=${DISPLAY_EVERY} --weak_scaling=true \
--parameter_server=${PS_SERVER} --device=gpu --variable_update=${VARIABLE_UPDATE}"
          OOM_LOG_MSG="OOM when allocating tensor"
        fi

        echo "Command to run: $BENCH_EXEC"
        echo "$BENCH_EXEC" >> $LOG_FOLDER/result.txt
        $BENCH_EXEC 2>&1 | tee $LOG_FOLDER/${GPUS}_${BIN_SEARCH_RUN}.txt
        ## Check of OOM occured
        echo "Test batch size ${BIN_SEARCH_RUN}" 
        ## Check of OOM occured
        if grep -q "${OOM_LOG_MSG}" $LOG_FOLDER/${GPUS}_${BIN_SEARCH_RUN}.txt
        then
            echo "Out of Memory at batchsize=${BIN_SEARCH_RUN}" | tee -a $LOG_FOLDER/result.txt
            RESULT=$BIN_SEARCH_RUN 
            BIN_SEARCH_HIGH=$BIN_SEARCH_RUN
            BIN_SEARCH_RUN=$(bc <<< "${BIN_SEARCH_HIGH}-(${BIN_SEARCH_HIGH}-${BIN_SEARCH_LOW})/2")
            RESULT_TEXT="oom"
        else
            # Not OOM keep tryig to find first OOM
            echo "Not OOM at batchsize=${BIN_SEARCH_RUN}" | tee -a $LOG_FOLDER/result.txt 
            BIN_SEARCH_LOW=$BIN_SEARCH_RUN
            BIN_SEARCH_RUN=$(bc <<< "${BIN_SEARCH_LOW}+(${BIN_SEARCH_HIGH}-${BIN_SEARCH_LOW})/2")
            RESULT_NEVER_PASS=false
        fi
        BIN_SEARCH_LAST=$BIN_SEARCH_RUN
        echo next move distance: $BIN_DISTANCE
    done

    if [ $RESULT_NEVER_PASS = "true" ]; then
        echo "Test was always OOM try a lower batch size" | tee -a $LOG_FOLDER/result.txt
    else
        echo "Final result batch-size: $RESULT ${RESULT_TEXT}" | tee -a $LOG_FOLDER/result.txt
    fi
    
fi 

