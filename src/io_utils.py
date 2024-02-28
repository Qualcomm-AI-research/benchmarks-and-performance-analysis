# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import datetime
import os 
import pathlib
from pathlib import Path
import pickle 
import sys 
import subprocess 
from typing import Any, Optional
import warnings 

SOURCE_DIRECTORY: pathlib.Path = pathlib.Path(__file__).parent.absolute()
PROJECT_DIRECTORY: pathlib.Path = pathlib.Path(SOURCE_DIRECTORY).parent.absolute()
SCRIPTS_DIRECTORY: pathlib.Path = PROJECT_DIRECTORY / "scripts"
LOG_DIRECTORY: pathlib.Path = PROJECT_DIRECTORY / "logs"
FIGURES_DIRECTORY: pathlib.Path = PROJECT_DIRECTORY / "figures"
DATA_DIRECTORY: pathlib.Path = PROJECT_DIRECTORY / "data"
BENCHMARK_DIRECTORY: pathlib.Path = pathlib.Path("/arm_benchmarks").absolute()
SIMULATION_LOGS_DIRECTORY: pathlib.Path = PROJECT_DIRECTORY / "simulation_logs"
CONFIGURATIONS_DIRECTORY: pathlib.Path = PROJECT_DIRECTORY / "system_configurations"
EXPERIMENT_LOGS_DIRECTORY: Path = PROJECT_DIRECTORY / "experiment_logs" 

for path in [SCRIPTS_DIRECTORY, LOG_DIRECTORY, FIGURES_DIRECTORY, DATA_DIRECTORY, SIMULATION_LOGS_DIRECTORY, CONFIGURATIONS_DIRECTORY, EXPERIMENT_LOGS_DIRECTORY]: 
    if not (path.exists() and path.is_dir()): 
        os.mkdir(path)

# gem5 directories 
GEM5_DIRECTORY: pathlib.Path = pathlib.Path("/gem5").absolute()
GEM5_CONFIGS_DIRECTORY: pathlib.Path = GEM5_DIRECTORY / "configs"
GEM5_ARM_BINARY_DIRECTORY: pathlib.Path = pathlib.Path("/arm_binaries").absolute()
GEM5_TEST_PROGRAM_DIRECTORY: pathlib.Path = GEM5_DIRECTORY / "tests/test-progs"

def setup_experiment_directory(name: str) -> Path: 
    now: str = get_now_str()
    parent_directory: Path = EXPERIMENT_LOGS_DIRECTORY / name 

    if not parent_directory.exists(): 
        parent_directory.mkdir(exist_ok=False)

    experiment_directory: Path = parent_directory / now 
    experiment_directory.mkdir(exist_ok=False)
    return experiment_directory

def get_now_str() -> str: 
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

def human_seconds_str(seconds: float) -> str:
    units: Tuple[str] = ("seconds", "milliseconds", "microseconds", "nanoseconds")
    power: int = 1

    for unit in units:
        if seconds > power:
            return f"{seconds:.1f} {unit}"

        seconds *= 1000

    return f"{int(seconds)} picoseconds"

def save_object(obj: Any, location: pathlib.Path):
    if location.absolute().as_posix()[-4:] != ".pkl":
        location = pathlib.Path(str(location) + ".pkl")
    with open(location, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_object(location: pathlib.Path) -> Any:
    with open(location, "rb") as handle:
        result = pickle.load(handle)
    return result

deserialize = load_object
serialize = save_object

def save_archive(source: pathlib.Path, compression: Optional[str]="gz"): 
    payload: pathlib.Path = source.with_suffix(f".tar.{compression}")
    run_status = subprocess.run(["tar", "-czf", payload.absolute().as_posix(), source.absolute().as_posix()], check=True)

def human_bytes_str(num_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB")
    power = 2**10

    for unit in units:
        if num_bytes < power:
            return f"{num_bytes:.1f} {unit}"

        num_bytes /= power

    return f"{num_bytes:.1f} TB"
