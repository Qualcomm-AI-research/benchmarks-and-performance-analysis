# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import jax.numpy as np 

def rms(x: np.ndarray) -> np.ndarray: 
    return np.linalg.norm(x) / np.sqrt(x.size)

def percentage_error(x: np.ndarray, reference: np.ndarray) -> np.ndarray: 
    return (rms(x - reference) / rms(reference)) * 100.0
