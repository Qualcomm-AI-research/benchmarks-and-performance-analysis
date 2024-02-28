# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import argparse 
from pathlib import Path 

import matplotlib.pyplot as plt 

from custom_logging import setup_logger
from io_utils import EXPERIMENT_LOGS_DIRECTORY

parser = argparse.ArgumentParser(description="Default experiment template.")

def main(args): 
    log = setup_logger(__name__)

if __name__=="__main__": 
    args = parser.parse_args()
    main(args)
