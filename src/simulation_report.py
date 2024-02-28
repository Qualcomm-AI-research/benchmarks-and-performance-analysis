# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import argparse 
from pathlib import Path 
from typing import Dict, List

import matplotlib 
import matplotlib.pyplot as plt 
import numpy as np 
from tabulate import tabulate

from custom_logging import setup_logger
from io_utils import deserialize, human_bytes_str, human_seconds_str

parser = argparse.ArgumentParser(description="Simulation report generation.") 
parser.add_argument("--results", required=True, type=str, help="Path to simulation output directory.")
parser.add_argument("--mpl-backend", default="Agg", type=str, help="Matplotlib backend.")
parser.add_argument("--plots", action="store_true", help="Render plots.")
parser.add_argument("--simulation-frequency", type=int, default=1_000_000_000_000, help="Ticks/second in the simulator.")
parser.add_argument("--sampling-period", type=int, default=10_000_000, help="Sampling period in ticks.")

def filter_nans(results: Dict[str, np.ndarray], **kwargs): 
    result_filtered: Dict[str, np.ndarray] = {} 

    total_nans: int = 0 
    average_nans: int = 0 
    num_with_nans: int = 0 

    for feature_id, timeseries in results.items(): 
        num_nans: int = np.count_nonzero(np.isnan(timeseries))
        if num_nans > 0: 
            num_with_nans += 1
            total_nans += num_nans
        average_nans += num_nans

        result_filtered[feature_id] = np.nan_to_num(timeseries, copy=True, nan=kwargs.get("nan", 0.0), posinf=kwargs.get("posinf", 0.0)) 

    return result_filtered, (total_nans, average_nans / len(results), num_with_nans)

def main(args): 
    # logging 
    log = setup_logger(__name__) 

    # load results 
    results_directory: Path = Path(args.results)
    results_path: Path = results_directory / "stats.pkl"
    results: Dict[str, np.ndarray] = deserialize(results_path) 
#    log.info(f"Loaded simulation results from: {results_path}")

    num_features: int = len(results)
    num_timesteps: int = list(results.values())[0].size
    program_duration: float = num_timesteps * (1 / args.sampling_period) 
#    log.info(f"Results contain {num_features} features with {num_timesteps} timesteps each ({human_bytes_str(num_timesteps * list(results.values())[0].itemsize)} of raw data).") 
    basic_results = [
            ("Results path", results_path), 
            ("Program duration", human_seconds_str(program_duration)), 
            ("Sampling period", human_seconds_str((1 / args.simulation_frequency) * args.sampling_period)), 
            ("Number of features", num_features), 
            ("Number of timesteps", num_timesteps), 
            ("Raw size", human_bytes_str(num_timesteps * list(results.values())[0].itemsize))
            ]
    log.info(tabulate(basic_results))

    # filter nans 
    results, (total_nans, average_nans, num_with_nans) = filter_nans(results)
    log.info("\n")
    log.info("---- NaN Info ---- ")

    nan_results = [
            ("Total NaN entries", total_nans), 
            ("Number of features with NaNs", num_with_nans), 
            ("Average number of NaNs per feature", f"{average_nans:0.1f}"), 
            ("Average percentage of NaNs per feature", f"{(average_nans / num_timesteps) * 100.0:0.1f}%")
            ]
    log.info(tabulate(nan_results))

    # TODO quantiles 



    # target timeseries 
    ipc_id: str = "system.cpu.totalIpc"

    if ipc_id in list(results.keys()): 
        log.info(f"results contained feature id {ipc_id}")
        ipc: np.ndarray = results["system.cpu.totalIpc"]
    else: 
        log.info(f"results did not containe feature id {ipc_id}. Computing manually.")

        num_instructions: np.ndarray = results["system.cpu.exec_context.thread_0.numInsts"]
        num_nonzeros: int = np.count_nonzero(np.isnan(num_instructions)) 
        if num_nonzeros > 0: 
            log.info(f"Number of instructions counter contained {num_nonzeros} NaNs... removing.")
            num_instructions = np.nan_to_num(num_instructions, copy=True, nan=0.0, posinf=0.0) 

        ipc: np.ndarray = num_instructions / results["system.cpu.numCycles"]

    # plots 
    if args.plots: 
        plt.figure() 
        plt.plot(ipc, c="tab:blue", label="IPC")
        plt.xlabel("Time")
        plt.ylabel("Instructions per cycle")
        save_path: Path = Path(results_directory / "ipc_timeseries")
        plt.savefig(save_path)
        plt.close()
        log.info(f"Saved IPC timeseries to: {save_path}.png")

if __name__=="__main__": 
    args = parser.parse_args()
    matplotlib.use(args.mpl_backend) 
    main(args)
