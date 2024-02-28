# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import argparse 
import os 
import subprocess 
import sys 

parser = argparse.ArgumentParser(description="Utility script to download and cross-compile a collection of benchmarks.")
parser.add_argument("--compiler", type=str, default="/usr/bin/aarch64-linux-gnu-gcc-9", help="Path to compiler.")
parser.add_argument("--disable_cache", )

def main(args: argparse.Namespace): 
    pass 

if __name__=="__main__": 
    args = parser.parse_args()
    main(args)
