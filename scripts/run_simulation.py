# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import argparse 
import glob 
import functools
import os 
import gzip 
import multiprocessing as mp 
import pathlib 
from pathlib import Path
import subprocess 
import sys 
import time 
from tqdm import tqdm 
from typing import List

from custom_logging import setup_logger
from io_utils import BENCHMARK_DIRECTORY, CONFIGURATIONS_DIRECTORY, DATA_DIRECTORY, SIMULATION_LOGS_DIRECTORY, SOURCE_DIRECTORY, get_now_str, save_archive

parser = argparse.ArgumentParser(description="Script to manage simulation/telemetry, and parsing of simulation outputs into raw serialized time series data.")

parser.add_argument("--sampling-period", default=1_000_000, help="Simulation sampling period (in units of ps).") 
parser.add_argument("--num-processes", default=3, type=int, help="Number of simulations to run concurrently.") 
parser.add_argument("--core-type", type=str, default="O3Cpu", choices=["O3Cpu", "ArmTimingSimple"], help="CPU model.")
parser.add_argument("--binaries", type=str, help="Path to directory with binary(ies) whose execution should be simulated.")

def simulate(args, binary: Path): 
    launch_timestamp: str = get_now_str()
    output_directory: pathlib.Path = SIMULATION_LOGS_DIRECTORY / pathlib.Path("multirun_out") / pathlib.Path(launch_timestamp)
    output_directory.mkdir(parents=True)
    log = setup_logger(__name__, custom_handle=Path(output_directory / "log.out").as_posix()) 

    configuration: pathlib.Path = CONFIGURATIONS_DIRECTORY / "arm_basic.py"

    simulation_command: List[str] = [
        "/gem5/build/ARM/gem5.opt", 
        "--outdir", 
        f"{str(output_directory)}", 
        f"{str(configuration)}", 
        "--cpu", 
        f"{args.core_type}", 
        "--binaries", 
        f"{binary.as_posix()}"
    ]

    # search the directory for any arguments to be passed to the binary 
    arguments: Path = Path(binary.parent / f"{binary.stem}.args")
    if arguments.exists(): 
        simulation_command.append("--binary-args")
        
        with open(arguments, 'r') as handle: 
            arguments = handle.read()
        simulation_command.append(f"{arguments.strip()}")

    if args.sampling_period: 
        simulation_command.append("--sampling-period")
        simulation_command.append(f"{args.sampling_period}")
    else: 
        simulation_command.append("--disable-sampling")


    parser_path: pathlib.Path = SOURCE_DIRECTORY / "parser.py"
    stats_path: pathlib.Path = output_directory / "stats.txt"
    compressed_stats_path: Path = output_directory / "stats.txt.gz"

    parse_command: List[str] = [
        "python3", 
        f"{str(parser_path)}", 
        "--stream", 
        "--num-blocks", 
        "500", 
        "--input", 
        f"{compressed_stats_path.as_posix()}", 
        "--remove-backup", 
    ]

    log.info(f"Launching simulation with command: {simulation_command}")
    simulation_status: subprocess.CompletedProcess = subprocess.run(simulation_command, check=True)
    log.info("simulation complete...")

    log.info("Compressing stats")

    with open(stats_path.as_posix(), 'rb') as input_handle: 
        with gzip.GzipFile(compressed_stats_path.as_posix(), 'wb') as compressed_handle: 
            compressed_handle.writelines(input_handle)

    log.info("Finished compressing stats")

    log.info("Removing uncompressed stats")
    stats_path.unlink()
    log.info("Removed uncompressed stats")

    log.info(f"Launching parsing with command: {parse_command}")
    parse_status: subprocess.CompletedProcess = subprocess.run(parse_command, check=True)
    log.handler_set = False 


def is_executable(path: Path) -> bool: 
    return path.is_file() and os.access(path, os.X_OK)

def get_executables(directory: Path) -> List[Path]: 
    executables: List[Path] = [] 
    for path in Path(directory).iterdir(): 
        if is_executable(path): 
            executables.append(path)
    return executables


def main(args: argparse.Namespace): 
    binaries: List[Path] = get_executables(Path(args.binaries))
    _simulate: callable = functools.partial(simulate, args) 

    for binary in binaries: 
        _simulate(binary)

if __name__=="__main__": 
    args = parser.parse_args()
    main(args)
