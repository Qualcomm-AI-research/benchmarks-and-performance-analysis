# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

from typing import Optional 

import numpy as np 

def least_squares(A: np.ndarray, b: np.ndarray, condition_threshold_db: Optional[float]=120, auto_normalize: Optional[bool]=False) -> np.ndarray: 
    if auto_normalize: 
        A = (A - A.mean(0)) / A.std(0)
        b = (b - b.mean(0)) / b.std(0)


    gram: np.ndarray = A.T @ A + np.eye(A.shape[1]) * 1e-3
    condition_number: float = np.linalg.cond(gram)
    breakpoint()
    
    if condition_number > (10**(condition_threshold_db * 0.05)): 
        raise ValueError(f"Provided coefficient matrix was singular (had condition number {(20 * np.log10(condition_number)):0.1f}dB larger than {condition_threshold_db}dB threshold)")

    return np.linalg.solve(gram, A.T @ b)
