#! /bin/bash
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

# Default values for optional arguments
BINARY="/gem5/tests/test-progs/hello/bin/arm/linux/hello"
SAMPLING_PERIOD=-1
PARSE=true

# Function to display usage message
display_usage() {
    echo "Usage: $0 <output-directory> [--binary=<binary_path>] [--sampling-period=<sampling_period>]"
}

# Check if --help option is provided
if [[ "$1" == "--help" ]]; then
    display_usage
    exit 0
fi

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --binary=*)
        BINARY="${key#*=}"
        shift
        ;;
        --sampling-period=*)
        SAMPLING_PERIOD="${key#*=}"
        shift
        ;;
        --debug*)
        PARSE=false
        shift
        ;;
        *)
        # Assume the argument is the directory name
        OUTPUT_DIRECTORY="$1"
        shift
        ;;
    esac
done

# If directory name is not provided, display usage message
if [ -z "${OUTPUT_DIRECTORY}" ]; then
    display_usage
    exit 1
fi

if [ -d "${OUTPUT_DIRECTORY}" ]; then
    echo "Output directory exists: ${OUTPUT_DIRECTORY}"
else
    mkdir -p "${OUTPUT_DIRECTORY}"
fi 

if [ "${SAMPLING_PERIOD}" -eq -1 ] ; then 
    /gem5/build/ARM/gem5.opt --outdir ${OUTPUT_DIRECTORY} configurations/arm_basic.py \
                                --stats-root "system.cpu.dcache" \
                                --stats-root "system.cpu.icache" \
                                --stats-root "system.cpu" \
                                --stats-root "system.l2bus" \
                                --disable-sampling \
                                --binary /arm_binaries/FloatMM 
else
    /gem5/build/ARM/gem5.opt --outdir ${OUTPUT_DIRECTORY} configurations/arm_basic.py \
                                --stats-root "system.cpu.dcache" \
                                --stats-root "system.cpu.icache" \
                                --stats-root "system.cpu" \
                                --stats-root "system.l2bus" \
                                --sampling-period=${SAMPLING_PERIOD} \
                                --binary /arm_binaries/FloatMM 
fi

if [[ ${PARSE} = false ]] ; then 
    exit 0 
else
    python3 src/parser.py --input "${OUTPUT_DIRECTORY}/stats.txt"
fi 
