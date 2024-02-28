# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import argparse 
import os 
from typing import Dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np 

from custom_logging import setup_logger
from io_utils import load_object, FIGURES_DIRECTORY, human_bytes_str

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, help="Path to serialized parsed data.")

def main(args): 
    if not os.path.exists(args.data): 
        raise FileNotFoundError(f"Did not find any data at: {args.data}")
    
    log = setup_logger(__name__)

    stats: Dict[str, np.ndarray] = load_object(args.data)
    log.info(f"Loaded stats from {args.data}")

    univariate_time_series: List[np.ndarray] = [] 
    keys: List[str] = [] 
    max_length: int = 0 

    for key, value in stats.items(): 
        if value.ndim < 2: 
            univariate_time_series.append(value)
            keys.append(key)

            if value.size > max_length: 
                max_length = value.size

    lengths: List[int] = [x.size for x in univariate_time_series]
    max_length: int = max(lengths)
    min_length: int = min(lengths)
    median_length: int = np.median(lengths)

    rms: callable = lambda ndarray: np.linalg.norm(ndarray) / ndarray.size

    log.info(f"Extracted {len(univariate_time_series)} univariate time series...")
    log.info(f"Minimum length: {min_length}")
    log.info(f"Maximum length: {max_length}")
    log.info(f"Median length: {median_length}")
    log.info(f"RMS length: {rms(np.array(lengths)):0.1f}")

    padded_time_series: List[np.ndarray] = [] 

    for time_series in univariate_time_series: 
        pad_width: int = max_length - time_series.size 
        if pad_width > 0: 
            time_series = np.pad(time_series, pad_width=(0, pad_width), mode='constant')

        padded_time_series.append(time_series)

    time_series: np.ndarray = np.vstack(padded_time_series)
    log.info(f"Padded time series occupy {human_bytes_str(time_series.nbytes)} in memory...")
    num_timesteps = max_length

    save_path: os.PathLike = os.path.join(FIGURES_DIRECTORY, "time_series_finegrained")

    smooth: callable = lambda ndarray, order: np.convolve(ndarray, np.ones(order) / order, mode='same')

    fig, axs = plt.subplots(nrows=3, ncols=1)

    max_y: int = None

    for key, x in zip(keys, padded_time_series): 
        axs[0].plot(np.arange(num_timesteps), x, label=key)

        if max_y is not None: 
            axs[0].set_ylim(0, max_y)

        smoothed: np.ndarray = smooth(x, 10)
        axs[1].plot(np.arange(smoothed.size), smoothed)

        if max_y is not None: 
            axs[1].set_ylim(0, max_y)

        smoothed: np.ndarray = smooth(x, 30)
        axs[2].plot(np.arange(smoothed.size), smoothed)

        if max_y is not None: 
            axs[2].set_ylim(0, max_y)

    axs[0].legend(loc="upper right")

    plt.savefig(save_path)
    plt.close()
    log.info(f"saved output to: {save_path}")

if __name__=="__main__": 
    args = parser.parse_args()
    main(args)
