#!/bin/bash
# monitor_gpus.sh - Monitor GPU usage during experiments

echo "Monitoring GPU usage (Ctrl+C to stop)..."
echo "Time | GPU | Util | Memory | Temp | Power | Process"
echo "====================================================="

while true; do
    timestamp=$(date +"%H:%M:%S")
    
    # Get GPU stats
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu,power.draw \
                --format=csv,noheader,nounits | while read line; do
        gpu_info=($line)
        gpu_id=${gpu_info[0]//,/}
        util=${gpu_info[1]//,/}
        mem=${gpu_info[2]//,/}
        temp=${gpu_info[3]//,/}
        power=${gpu_info[4]//,/}
        
        # Check if ga_solver is running on this GPU
        process=$(nvidia-smi -i $gpu_id --query-compute-apps=pid,name --format=csv,noheader | grep ga_solver | head -1)
        
        printf "%s | %3s | %3s%% | %5sMB | %3s°C | %5sW | %s\n" \
               "$timestamp" "$gpu_id" "$util" "$mem" "$temp" "$power" "$process"
    done
    
    sleep 5
done