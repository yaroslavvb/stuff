#!/bin/bash

./monitor_nvidia.sh --log_full_path ./full_log.txt --log_summary_full_path ./log_summary.txt  &

NVIDIA_MONITOR=$!

echo "Log monitor pid ${NVIDIA_MONITOR}"

sleep 12

kill $NVIDIA_MONITOR
wait $NVIDIA_MONITOR
echo "Success:  ${NVIDIA_MONITOR} is not longer running" 

# put this in any script to hard kill monitor_nvidida
echo "Test killing with pgrep"
./monitor_nvidia.sh --log_full_path ./full_log --log_summary_full_path ./log_summary  &
NVIDIA_MONITOR_2=$!

sleep 12

echo "Log monitor pid ${NVIDIA_MONITOR_2}"
echo "kill with pgrep"
pgrep "monitor_nvidia" | xargs kill

echo "Wait until dead"
wait $NVIDIA_MONITOR_2
echo "Process is dead:  Test successful"

