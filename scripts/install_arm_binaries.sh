#! /bin/bash 
# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

mkdir -p /arm_binaries && cd /arm_binaries
wget -O Bubblesort dist.gem5.org/dist/v22-0/test-progs/cpu-tests/bin/arm/Bubblesort
wget -O FloatMM dist.gem5.org/dist/v22-0/test-progs/cpu-tests/bin/arm/FloatMM
