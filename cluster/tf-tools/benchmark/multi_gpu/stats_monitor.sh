#!/bin/bash


# Get all nvidia-smi data worth having
# There is no historical data so calling this after a run
# when the GPU may no long be throttled is of no value.

KEEP_LOOP=true
LOG_FULL_PATH="./monitor_log.txt"
LOG_SUMMARY_FULL_PATH="./log_summary.txt"


while [[ $# -gt -0 ]]; do
  key="$1"
#  echo $key
#  echo $2
  case $key in
    --log_full_path)
      LOG_FULL_PATH="$2" # location to log raw monitoring logs, e.g. nvidia-smi
      shift
      ;;
    --log_summary_full_path)
      LOG_SUMMARY_FULL_PATH="$2"  # Format of the data NHWC or NCHW (NVIDIA)
      shift
      ;;
    *)
      echo "Unknown flag: $key"
      ;;
  esac
  shift # past argument or value
done




MAX_SLOWDOWN_GPUS=0
# Handle CTRL-C or other term signal, log the max number of GPUs that showed
# Errors, for now that means "HW Slowdown: Active"
function summarizeCleanup {
  echo "Max GPUs throttled: ${MAX_SLOWDOWN_GPUS}"
  echo "Max GPUs throttled: ${MAX_SLOWDOWN_GPUS}" >> $LOG_SUMMARY_FULL_PATH
  exit;
}

# catch being asked to end
trap summarizeCleanup SIGINT SIGTERM

# Log nvidia-smi data forever (until killed externally) and track when HW Slowdown: Active occures 
# which indicates overheating and likely lower clock.
while [ "$KEEP_LOOP" = "true" ]; do

  RESULT=$(nvidia-smi -q -d UTILIZATION,CLOCK,PERFORMANCE | tee -a ${LOG_FULL_PATH} | \
grep -E 'HW Slowdown' | awk '!/Not Active/ {count++} END{print count}')
  
  # Handle result being blank.  Likely a better way using awk above
  if [ "$RESULT" = "" ]; then
    RESULT=0
  fi  

  if [ "$RESULT" -gt "$MAX_SLOWDOWN_GPUS" ]; then
  	MAX_SLOWDOWN_GPUS=$RESULT
  	echo "$MAX_SLOWDOWN_GPUS GPU(s) with slowdown"
  fi

  
  # 10 second seem to be reasonable.
  sleep 10

done

