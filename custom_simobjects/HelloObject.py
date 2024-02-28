# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

from pathlib import Path 

from m5.params import * 
from m5.SimObject import SimObject 

class HelloObject(SimObject): 
    type = "HelloObject" 
    cxx_header: Path = "/forecaster/custom_simobjects/include/hello_object.hh" 
    cxx_class: str = "gem5::HelloObject" 
