# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import argparse
from pathlib import Path 

from custom_logging import setup_logger

parser = argparse.ArgumentParser(description="A generic Python script template.")

def main(args): 
    log = setup_logger(__name__) 
    pass

if __name__=="__main__": 
    args = parser.parse_args()
    main(args)
