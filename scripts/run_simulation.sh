#! /bin/bash 
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

# 
# Description: convenience script to launch simulations 
# Author: nichrich@qti.qualcomm.com

BINARIES="/forecaster/bin" 
SAMPLING_PERIOD=1000000
MAX_CONCURRENT=5

show_usage() {
    echo "Usage: $(basename "$0") [options]" 
    echo "Options:"
    echo "  --binaries <directory>      Set the binaries directory (default /forecaster/bin)."
    echo "  --sampling-period  <n>      Set the simulation telemetry sampling period (default 1000000, or 1us)."
    echo "  --max-concurrent   <n>      Set the maximum number of concurrent simulations to launch."
    echo "  -h, --help                  Show this usage message."
}

while [[ $# -gt 0 ]]; do 
    case "$1" in
        --binaries)
            BINARIES_DIR="$2"
            shift 2 
            ;;
        --sampling-period)
            SAMPLING_PERIOD="$2"
            shift 2 
            ;;
        --max-concurrent)
            MAX_CONCURRENT="$2"
            shift 2
            ;;
        -h|--help|help)
            show_usage 
            exit 0 
            ;;
        *)
            echo "Error: Unrecognized argument '$1'"
            show_usage 
            exit 1 
            ;;
    esac
done

mkdir tmp
touch tmp/launch.out 

for file in "${BINARIES}"/*; do 
    if [[ -x "$file" ]]; then 
        echo "python3 /forecaster/scripts/run_simulation.py --binaries $(realpath ${file}) --sampling-period ${SAMPLING_PERIOD} > /forecaster/us_sampling_logs/$(basename ${file}).out 2>&1 &" >> tmp/launch.out
    fi 
done 

#cat tmp/launch.out 
#rm -rf tmp 
